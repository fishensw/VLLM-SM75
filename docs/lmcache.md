# LMCache 分层缓存（v0.1.6 实验功能）

本候选接入 LMCache MP：GPU 保留当前计算需要的 KV，CPU 保存可复用前缀，专用磁盘保存更大的缓存集合。重复长提示、公共系统提示或恢复会话时，可从较低层把匹配的 KV 搬回 GPU，减少重复预填充。磁盘层的收益与传输开销取决于命中率、CPU/PCIe 和存储速度；首次请求仍需正常计算。

CPU 与磁盘都不参与本实现的 GPU 注意力计算，容量不能直接加到单个活动请求的 GPU KV 上限。YaRN 负责位置长度扩展，LMCache 负责缓存复用，两者均不代表小显存已经可以运行 1M 上下文。

完整Ultra候选已通过四T10的八项整模型CPU/磁盘/重启回读复验，详情见[分阶段验收](validation/v0.1.6-lmcache.md)。范围为target-only的8K合成文本，不扩展为1M或投机组合的保证。

## 固定运行环境

- vLLM 0.30.0 / Python 3.12 / Linux x86_64 / torch 2.13.0+cu129。
- LMCache 0.5.5 cu129，固定 wheel SHA256；CuPy 使用 cuda12x 14.2.0，清理不同 CuPy 包共享同一命名空间的冲突。
- 必须带本候选回移的 [PR #4731](https://github.com/LMCache/LMCache/pull/4731) packed4D 修复。此 PR 在 2026-09-23 仍未合并；本项目回移新规则并适配 vLLM 0.30 的逻辑轴约定，不改变旧的 5D/MLA/Mamba 规则。
- 实测原版 0.5.5 在 Qwen3.8-27B、TP4、FP8 KV 上漏传逻辑页内容：首次回答正常，从 CPU 恢复后乱码。原样回移 PR 后还会因逻辑轴/物理布局混淆被校验拦截；本地适配先核对真实规格，再做无复制视图转换。安装器校验原始/修补文件 SHA，受管启动探针拒绝未修补或未知版本，不能用普通 `pip install lmcache` 替代。

补丁、来源、Apache 许可证与精确哈希位于 [patches](../ultra/source/lmcache/patches/README.md)，依赖在 [install-lock.json](../ultra/source/lmcache/install-lock.json)。安装器不解析升级 torch、vLLM、CUDA 或 FlashInfer；现有依赖不满足固定要求会先报错。旧基座自带的 LMCache 0.5.4 先按安装 RECORD 检查原目标文件，再升级；未知版本、被修改或未登记的目标文件不会被直接覆盖。

## 安装入口

本项目固定的cu129扩展默认不额外安装、不启用。官方基座可能自带未适配的LMCache；本轮完整标准版自带0.5.5，但其CUDA13依赖声明与固定cu129栈不符，不能直接启用，必须先安装下面的固定扩展。构建 Ultra 时显式添加 `--build-arg INSTALL_LMCACHE=1`；现有全量构建保护仍生效。该选项同时适用于 Ultra 的完整 Dockerfile 和增量候选 Dockerfile。

已有**经过验证的 v0.1.6 镜像**可以使用单独扩展层（从仓库根目录构建）。该层只安装 LMCache；Ultra 控制台的新功能仍需本候选的 console 源码，不能仅给旧 UI 镜像加此层：

```bash
docker build -f docker/Dockerfile.lmcache \
  --build-arg BASE_IMAGE=<已固定的v0.1.6镜像或摘要> \
  -t local/vllm-sm75:v0.1.6-lmcache .
```

直接检查独立测试容器的安装计划：

```bash
python3 /opt/sm75-workbench/lmcache/install.py
# 仅在明确要安装扩展的候选环境执行：
python3 /opt/sm75-workbench/lmcache/install.py --install
```

标准扩展镜像的安装器路径为 `/opt/vllm-sm75/lmcache/install.py`。完整镜像构建和安装器在已有测试环境内运行是不同的验收项；见后附记录。

## 控制台配置

运行配置 → 编辑 → **CPU KV 与分层缓存** → 缓存方案选择 **LMCache**。选择“默认”显示引擎原生 CPU KV 选项，选择“LMCache”显示 CPU 与磁盘分层配置；端口和磁盘门槛在“磁盘与服务设置”中展开。

| 配置项 | 含义 |
| --- | --- |
| CPU 缓存总容量 | 整个 MP 服务的 GiB，总量只设一次，不乘 TP 卡数 |
| Chunk size | 必填；按实际模型、TP、KV 精度及混合层布局的启动日志设置 |
| 磁盘缓存目录 | 容器内专用绝对目录，建议挂载独立持久卷；留空只用 CPU |
| 启动所需剩余空间 | 启动前可用空间门槛；不是磁盘容量限额 |
| RPC / HTTP 端口 | 默认 5555 / 5556，均只监听容器内 localhost，不应对外发布 |

选择 LMCache 后点击顶部“保存配置”；验证成功时替换已知原生 CPU KV，无需逐项点击应用。自定义连接器由用户自行移除。配置保留原参数和草稿，遇到不兼容组合会报错，不静默删除。当前实验范围拒绝 DFlash/MTP、释放显存的休眠、自动休眠和 `expandable_segments`；使用 P-State 电源模式。

保存、模板、导入导出均保留 LMCache 配置。保存不会重启模型，下次启动才生效。启动时检查版本/补丁、目录、剩余空间、端口，然后等待真实健康接口就绪再启动模型；依赖检查失败会保留当前模型。停止模型会连同缓存服务停止，释放 CUDA IPC 引用；CPU L1 随服务退出丢失，已完成的磁盘缓存保留。

受管启动支持 Ultra 单容器与 Linux native。普通独立 Docker profile 不会自动在容器外启动一个无法共享 CUDA 内存的服务，需按下文在同一容器内配置。

### Chunk size 不能照抄

本次 Qwen3.8-27B、TP4、FP8 KV、关闭投机的实际统一块为 **1568 tokens**，一个逻辑块包含 49 个 32-token 内核页。官方单卡例子的 784、此前带草稿运行的 1664 均不可替代这个配置。

换模型、TP、KV 精度或注意力后端后，应重新核对 `Setting attention block size to N tokens` 及所有组的共同粒度；chunk 必须是共同粒度的倍数。预填充批次至少为 N，设为 `2N-1`（本例 3135）便于逐块快照并让解码与预填充并行调度。vLLM/连接器仍执行自身的容量与粒度检查，工作台不绕过它们。

## 标准版的同容器启动示例

先在模型容器内启动 MP 服务。目录中的 `layout-id` 需由操作者为具体权重、TP、KV 精度和引擎/后端版本单独命名，不可跨布局复用；Ultra 会根据模型元数据、参数、布局环境和安装源码指纹自动生成子目录。

```bash
python3 -m lmcache.v1.multiprocess.http_server \
  --host 127.0.0.1 --port 5555 \
  --http-host 127.0.0.1 --http-port 5556 \
  --no-isolated-ipc --chunk-size 1568 --separate-object-groups \
  --l1-size-gb 8 --l1-init-size-gb 1 --eviction-policy LRU \
  --l2-adapter '{"type":"fs","base_path":"/lmcache/layout-id","use_odirect":false}'
```

待 `/healthcheck` 返回 `status: healthy`，在同一容器的模型启动参数中加入：

```bash
--enable-prefix-caching --mamba-cache-mode align \
--max-num-batched-tokens 3135 \
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5555,"lmcache.mp.isolated_ipc":false}}'
```

这只是缓存参数，完整模型、GPU 和内存预算沿用对应模型的已验证配置；不能直接给任意模型套用 1568。停止顺序为先模型、后 MP，等待已完成写入后再做正常重启。后台进程必须由现有进程管理器负责退出与日志。

## 磁盘与兼容边界

本候选选择 Python `fs`，避开 0.5.5 `fs_native` 对带 salt 的混合组缓存键的兼容问题。默认存储策略保留 CPU 副本并异步写磁盘，磁盘可跨正常进程重启复用；并未承诺掉电持久性，进行中的写入也可能尚未落完。

`fs` 没有硬容量上限；需要硬限制时，应使用带文件系统配额的专用卷。目录不是编译缓存目录，也不由“清编译缓存”操作清除。普通文件读取可能来自操作系统页缓存，所以磁盘层命中不代表测到了物理盘吞吐量。

只验收文本路径；Qwen 混合 GDN 页不启用 CacheBlend 或 CacheGen，不据此宣称图片/视频 KV 缓存、跨主机共享、分布式缓存或投机组合可用。GDN 的冷算与恢复不保证逐 token 恒等，验收同时检查任务答案、命中来源和必要的字节传输完整性。

- [本轮测试与已知限制](validation/v0.1.6-lmcache.md)
- [官方 vLLM 兼容说明](https://docs.lmcache.ai/getting_started/compatibility.html)
- [官方 Qwen 混合模型说明](https://docs.lmcache.ai/recipes/qwen3_5.html)

构建环境没有 GPU 时，Dockerfile 使用 `--build-no-gpu`，仍加载两个原生扩展检查 ABI，但将需要 `libcuda.so.1` 的 MP 连接器导入留给 GPU 运行时。Ultra 启动 LMCache 前必须通过 `lmcache_probe.py` 的完整检查；镜像安装成功不等于分层缓存已经通过 GPU 验收。
