# v0.1.5 / ultra 整理与回归审计

## 基线与源码继承

远端 HEAD 只读核对仍为 95d3bcd168d14f1a0246878f80472e16826ea731（公开 v0.1.4 源码）。服务器保留标签 vllm-sm75:v0.1.4 的镜像 ID 为 9ea6136cc49c…，与历史文档的 146639… 不同，因此称为“保留 v0.1.4 镜像”，不冒充历史实测镜像。

本轮无 GPU 的镜像源文件审计：标准版与当前 ultra 各 43/43 匹配当前受审源码，注册 hook 均存在。37/43 项与保留 v0.1.4 镜像逐字节一致；差异为 monitor、dashboard、envs_sm75、scheduler_sm75、async_llm 和 engine/core。具体哈希见 [机器可读三方对照](2026-09-18-runtime-inheritance.json)。

| v0.1.4 优化 | 当前继承证据 | 性能结论边界 |
|---|---|---|
| Firefly AWQ INT4 prefill | CUDA/Python 与 Marlin 路由一致；默认开关保留 | AWQ 同契约性能仍需覆盖 |
| FP8 all-reduce | CUDA/Python、cuda_communicator 一致；默认 auto、1 MiB 门槛保留 | 需记录 P2P/SHM/NCCL 实际后端 |
| FlashQLA GDN prefill、Triton decode | vendored 源码、GDN 路由与 FlashInfer backend 一致 | 完整候选已验收，当前性能另测 |
| DFlash2/MTP | 8 项 speculative overlay 一致，激活 hook 保留 | 新动态投机 scheduler 继承 AsyncScheduler；不能误用同步调度 |
| CPU KV / prefix cache | connector、cache dtype 与分组相关实现一致 | 历史长上下文实际恢复通过，新增性能采样禁止命中 |
| 休眠与编译缓存 | auto_sleep/core_client/compilation 保留；envs 改为幂等扩展注册 | 独立冷编译 token 稳定性门禁仍在 |

## 工程收敛

公共入口统一为 build.sh/run.sh；版本号只从 VERSION 获取。删除当前目录中的 v0.1.3/v0.1.4 Dockerfile、BUILD 副本、根目录重复安装器和失效测试。fast 配方改为无版本名，复用 helpers 安装器；运行中不得对生产容器执行 fast 编译。

ultra 首次安装默认 cacheRoot 从未持久化的 /cache 改为 /data/cache（root/cache）；已有保存设置优先，保持原目录。标准运行入口明确 NVCC_THREADS=1，可覆盖，避免之前编译契约不一致导致的输出哈希差异。其余推理参数与开关保持原值。

新增命令构造测试用模拟 Docker 覆盖 standard/full、ultra/full、fast、ui、无效参数、优化开关和 ultra 权限/持久挂载，不操作真实容器。新增缓存默认值/旧配置保留测试。

## 验证与发布边界

本轮构建/启动契约 15 项、启动 shell 场景 15 项、休眠缓存测试 2 项、性能比较器测试 3 项、长上下文客户端测试 4 项、OCI 导入测试 2 项通过；Windows 控制台 51 通过、2 Linux 专用项跳过。当前控制台源码另外在无 GPU、768 MiB/1 CPU 的独立 Linux 容器中完成 53/53 测试。测试通过不冒充最终镜像已构建；最终源码仍须新镜像构建和启动验收。

已完成的有界性能契约：保留 v0.1.4、现有 v0.1.5、线上 ultra；27B FP8+DFlash2 TP4，8K/32K 输入、512 输出、greedy、每长度独立预热+5 次、每请求新 cache_salt。记录首 token 时间、估算 prefill、decode、TPOT、输入/输出 token hash、cache 命中、抢占、检索结果。ignore_eos=true 仅用于固定长度计时，不是自然停止质量测试。所有测量必须有零缓存命中、零抢占和正确检索；结论以最终 JSON 为准。

已有 262144 上下文与 512 输出验收是功能证据，不能替代性能对照；AWQ、MTP、target-only、多模型、自然停止与监控开销矩阵仍独立列为门禁。生产热修复不等同于可复现发布镜像，不会推送 GitHub 或公开镜像。

## 可重复的性能条件

本轮测量使用原有 4×T10（TP4）和相同 target/draft 文件、tokenizer，服务别名保留 `VLLM-Qwen3.5-27B`。仓库部署名称仍为 Qwen3.8-27B-FP8；命名并不能替代权重版本审计，权重 revision 尚未补齐。

- FP8 KV `fp8_e4m3`，最大上下文 262144，seq4 / batch8192 / block32。
- 本次生产契约为 GPU utilization 0.92、KV bytes 3288334336；不是统一启动脚本默认值的性能证明。
- DFlash draft 7、draft TP4、probabilistic sample、FLASHINFER；CPU KV 8589934592 bytes。
- `FULL_AND_PIECEWISE`、capture sizes `[8]`、async scheduling；v0.1.4 使用原 AsyncScheduler，v0.1.5/ultra 使用继承它的 SM75Scheduler。
- standard/v0.1.4 使用同一隔离编译缓存、NVCC_THREADS=1；ultra 使用生产持久缓存。不是独立冷编译稳定性证明。
- 测量顺序 ultra → v0.1.4 → v0.1.5，每长度先预热一次再连续五次，不随机交错；GPU 为驱动自动时钟，未锁定固定频率。两个标准容器记录 P-State/时钟/功耗/显存及宿主内存，生产 ultra 未同步采集同等逐时 GPU 记录，因此不能声称严格控制了全部硬件漂移。
- ultra 测量包含当前控制台与监控；它和标准版的差值不能单独当成监控组件因果开销。

本轮使用独立 network-none canary，未挂载生产数据目录，仅临时停止原模型容器；退出陷阱恢复原容器和 DSH。Docker daemon 不停止。原生产容器的静态热修复保持在原容器中，不用旧 RC8 替代。

## 三方实测结果

[原始 30 个样本、哈希、镜像 ID 与比较结果](2026-09-18-paired-performance.json)。三方各 10 个有效样本，全部完成 512 输出、零缓存命中、零抢占、检索通过；同输入的输出 token SHA256 在三个版本与所有重复之间一致。

以下均为五次中位数，P/D 单位 tok/s：

| 版本 | 8K P | 8K D | 32K P | 32K D | D 相对 v0.1.4（8K / 32K） |
|---|---:|---:|---:|---:|---:|
| 保留 v0.1.4 | 1268.73 | 63.65 | 1203.47 | 60.11 | 基线 |
| v0.1.5 标准 | 1267.09 | 63.60 | 1202.19 | 60.10 | −0.09% / −0.02% |
| 当前 ultra | 1271.23 | 63.43 | 1203.56 | 59.95 | −0.35% / −0.26% |

本契约通过预设 5% 吞吐门槛，未观察到明显劣化；最大中位数吞吐下降约 0.35%。这不是零开销证明，也不是所有模式或新构建镜像的性能认证。v0.1.4/standard 采样显存峰值均为每卡 15277 MiB；包含启动阶段，并非精确请求峰值。

恢复验收：测试包装脚本退出码 0，原生产控制台、DSH、模型 health 均恢复；控制台与引擎 HTTP 200。Docker daemon PID 始终为 26579。两个测试 canary 已停止，原回滚容器和镜像保留。
