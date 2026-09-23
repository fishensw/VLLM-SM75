# v0.1.6 参数推荐模板

示例按 Qwen3.8-27B-FP8、4×T10 16 GiB、TP4 编写。先按 [README](../README.md) 准备模型、缓存、API key 和 NVAPI 路径。每次只运行一个 GPU 服务；不同配置使用不同容器名，脚本不会替你停止现有容器。下表描述预期用途，实测性能与最大长度见 [验收报告](validation/v0.1.6-full-images.md)。

## 场景与效果

| 场景 | 核心配置 | 预期效果与代价 |
| --- | --- | --- |
| 首次验证 / 日常短上下文 | base、32K、seq4、batch8192、CPU KV 2 GiB | 不加载草稿，便于确认目标模型和容量；无投机加速 |
| DFlash2 对照 | draft7、256K、seq4、batch8192、每卡 KV 3,288,334,336 bytes、CPU 8 GiB | 对齐既有正式 FP8 配置；收益取决于接受率，草稿占用额外显存 |
| 单用户长上下文 | base、seq1、batch4096、GPU 自动预算 | 去掉草稿并降低并发/预填充峰值，为活动 KV 留空间；仍需真实容量检查 |
| 多用户 | 较短 max-model-len、seq8、batch8192 起步 | 增加调度并发；同样 GPU KV 由多个请求共享，单请求速度可能降低 |
| 公共长前缀复用 | 原生 CPU KV 或 LMCache 二选一 | 重复提示可减少 prefill；冷请求不因缓存容量增加而变快 |
| 常驻低延迟省电 | P-State | 权重/KV 常驻，空闲低频；不释放显存 |
| 闲置释放显存 | sleep / exit、30 分钟 | 引擎退出释放显存，下一请求需重新加载；编译缓存可复用 |

## 标准版可覆盖参数

| 环境变量 | 单位 / 默认 | 说明 |
| --- | --- | --- |
| `MAX_MODEL_LEN` | token；base/mtp 为 auto，FP8 DFlash2 为 262144 | 输入＋输出总长度；不是输入长度 |
| `MAX_NUM_SEQS` | 请求数；FP8 4、AWQ 8 | 同时调度上限 |
| `MAX_NUM_BATCHED_TOKENS` | token；FP8 8192、AWQ 16384 | 每步 prefill/decode 预算；不是最大上下文 |
| `GPU_MEMORY_UTILIZATION` | 比例；0.87，FP8 DFlash2 0.92 | 未显式设 GPU KV bytes 时参与自动容量规划 |
| `GPU_KV_BYTES` | bytes / GPU；DFlash2 FP8 3288334336，AWQ 4294967296 | 显式 KV 配额覆盖自动 KV 预算；0 表示不传固定配额 |
| `CPU_KV_GIB` | GiB / 整个 TP 服务；8 | 启动脚本要求整数；原生 CPU 前缀缓存，0 关闭。Ultra 面板可填写小数。宿主须另留权重加载、进程和系统内存 |
| `HF_OVERRIDES` | JSON；`{"dtype":"float16"}` | YaRN 等模型覆盖，保留必要 dtype；不同模型字段不可照抄 |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | 0 / 1；默认 0 | Marlin 归约方式；本轮正式参数对照为 1。变更后重新验收速度与输出 |
| `VARIANT` | base / mtp / dflash2 | MTP 需匹配权重；DFlash2 必须配置 `DRAFT_MODEL` |
| `POWER_MODE` | pstate / sleep；pstate | 与下文电源变量配套 |

`bash docker/run.sh` 末尾也可附加原生 vLLM 参数；配置同一项时优先使用上表变量，避免重复参数。该启动脚本固定 TP4，其他拓扑应使用经过核对的原生启动配置或 Ultra 参数编辑器。

## 模板 A：32K 基础推理

```bash
CONTAINER_NAME=sm75-base-32k VARIANT=base FORMAT=fp8 \
  MAX_MODEL_LEN=32768 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=8192 \
  CPU_KV_GIB=2 POWER_MODE=pstate bash docker/run.sh
```

CPU KV 2 GiB 作为复用层起步，不改变 GPU 活动 KV 容量。确认基本推理后再增加上下文与并发。

## 模板 B：正式 DFlash2 参数对照

```bash
CONTAINER_NAME=sm75-dflash2-256k VARIANT=dflash2 FORMAT=fp8 \
  DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2 \
  MAX_MODEL_LEN=262144 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=8192 \
  GPU_KV_BYTES=3288334336 CPU_KV_GIB=8 POWER_MODE=pstate \
  VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  bash docker/run.sh --mamba-cache-mode align \
  --no-disable-hybrid-kv-cache-manager --enable-prompt-tokens-details
```

此模板覆盖核心推理参数。本轮隔离基准还固定了 NCCL 通信环境、容器 CPU/RAM 限额和编译设置；精确复测需同时核对 [结构化记录中的 measuredDflashRuntime](validation/v0.1.6-full-images-results.json)，不能默认任意宿主得到相同速度。

脚本设置 draft7、draft TP4、草稿上限 262144、FLASHINFER、probabilistic 采样与 graph capture size 8。Ultra 在参数编辑器中使用相同值，并在环境变量中设置 `VLLM_MARLIN_USE_ATOMIC_ADD=1`。这是本轮对照条件，尚未通过关闭该开关来证明跨进程 token 差异的根因。GPU 配额是每卡约 3.06 GiB，CPU 8 GiB 是整组总量。CPU KV 和 DFlash/GDN 的共同可恢复前缀存在限制，实际回读看计数器；有配置不代表有命中。本轮完整标准版的 261632 输入＋512 输出两次均通过、零抢占，但重复请求没有 CPU 回读，TTFT 仍约 336 秒。不要用这套配置承诺热前缀加速。

本轮额外测试 draft3：同样 KV 配额下无法启动 256K；目标/草稿均降为 160K 后，128K decode 为 52.85 tok/s，没有改善，因此仍保留 draft7 对照模板。

## 模板 C：长上下文与 YaRN 1M 配置

先关闭投机，seq1、batch4096，按 128K → 256K → 更长逐级验证。`GPU_KV_BYTES=0` 使用自动 KV 预算；减小 GPU KV 并不会增加单请求最大长度。

以下仅针对原生文本长度为 262144 的本次 Qwen 复合配置：

```bash
export HF_OVERRIDES='{"dtype":"float16","text_config":{"max_position_embeddings":1048576,"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}'
CONTAINER_NAME=sm75-yarn-capacity VARIANT=base \
  MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=4096 \
  GPU_KV_BYTES=0 GPU_MEMORY_UTILIZATION=0.92 CPU_KV_GIB=0 \
  POWER_MODE=pstate bash docker/run.sh --mamba-cache-mode align
```

vLLM 0.30 深度合并 text_config，保留模型原生 mRoPE、theta 和 partial rotary 字段。仅写 factor 或写到错误的顶层不足以开启目标长度。不设置绕过长度验证的环境变量。若 GPU KV 容量检查拒绝 1M，减少目标长度或增加实际可用 GPU 容量；CPU KV/LMCache 无法替代这部分容量。

容量探索时可将上例 `MAX_MODEL_LEN=1048576` 改为 `MAX_MODEL_LEN=auto`：vLLM 会根据所有 GPU 的实际缓存分组和保留块自动下调到可容纳的长度，最终可用值以启动日志为准。纯文本场景还可追加 `--language-model-only`，释放视觉部分的预算，同时关闭图片输入能力；它与关闭 DFlash 是两个独立选择。换成 AWQ 可减少权重占用，但属于另一组模型/量化配置，必须单独验证质量和长度。

Ultra：运行配置 → 上下文与 KV → 读取原生长度 → 开启 YaRN → 填 1048576；单独设置 GPU/CPU KV，再检查高级参数。草稿有自己的 RoPE 和最大长度，主模型覆盖不会自动传给草稿；超出已验证草稿上限时关闭 DFlash/MTP。

`auto` 适合探索引擎容量；当前 Harness 配置生成器在此模式下不写入明确的 `contextWindow`。需要工作台收到确定预算时，将运行配置改为本机已验证的总长度并重启，再检查 Harness 模型设置；不会仅因 YaRN 目标填了 1M 就认定工作台拥有可用的 1M 预算。

较长请求还需核对客户端和反向代理的读取/空闲超时；配置目标长度前应先确认实际 GPU KV 容量。

## 模板 D：LMCache 复用

构建 Ultra 时使用 `EDITION=ultra INSTALL_LMCACHE=1 bash docker/build.sh`，安装后仍默认关闭。在“快速组合与投机解码”将加速设为“无”并应用，电源选择 P-State；在“CPU KV 与分层缓存”将缓存方案选为 LMCache，保存时替换原生 CPU KV。关闭自动休眠和 `expandable_segments`；配置 CPU 总 GiB、共同 chunk 粒度及专用磁盘路径。切换连接器需要重新启动模型。

本次 TP4、FP8 KV、target-only 8K 验证的 chunk 是 1568，可使用 batch3135。此值来自混合页布局，换模型/TP/精度必须重新核对。磁盘空间门槛不是硬配额；需要容量限制时使用带文件系统配额的专用卷。详见 [安装与标准版同容器示例](lmcache.md)。

## 电源模式与单位

| 项目 | 建议起点 | 效果 |
| --- | --- | --- |
| `POWER_MODE=pstate` | 常驻服务 | 关闭自动显存休眠；保持模型和 KV |
| `PSTATE_IDLE_TIMEOUT=1800` | **秒** | 空闲计时；正式旧配置可能是 1 秒，导入后核对 |
| `PSTATE_CONFIRM=60` | **秒** | 低负载持续确认时间，减少频繁切换 |
| `PSTATE_UTIL=5` | GPU 利用率百分比 | 低负载判定门槛 |
| `PSTATE_LOW=8` / `PSTATE_HIGH=16` | 8 / 16 | P8 与恢复驱动自动控制；禁止误解为固定 P16 |
| `PSTATE_POLL=5` | **秒** | 状态采样间隔 |
| `PSTATE_GPUS=0,1,2,3` | 容器可见 GPU 序号 | 只管理本服务的 GPU |
| `POWER_MODE=sleep` | 闲置服务 | 使用下面的自动休眠参数 |
| `AUTO_SLEEP_IDLE_TIMEOUT=30` | **分钟** | 自动休眠等待；0 关闭 |
| `AUTO_SLEEP_OFFLOAD_TARGET=exit` | exit | 退出引擎；下次请求重载，不保留对话 KV |

```bash
# 常驻；需要匹配驱动的 PSTATE_NVAPI_LIB
POWER_MODE=pstate PSTATE_IDLE_TIMEOUT=1800 PSTATE_CONFIRM=60 bash docker/run.sh
# 显存释放；无需 P-State 控制器
POWER_MODE=sleep AUTO_SLEEP_IDLE_TIMEOUT=30 AUTO_SLEEP_OFFLOAD_TARGET=exit bash docker/run.sh
# 无 NVAPI 的基础性能测试：既不管理 P-State，也不自动休眠
POWER_MODE=sleep AUTO_SLEEP_IDLE_TIMEOUT=0 bash docker/run.sh
```

P-State 模式忽略 `AUTO_SLEEP_*`；Ultra 的电源计时在每份配置中保存，单位以面板标签为准。`cpu/reload/disk` 休眠需要额外容量与兼容性验证，见 [休眠指南](sleep-and-cache.md)。LMCache 实验路径只允许保持显存的 P-State。

当前 Ultra 的“导入配置”入口重建本机配置时不带入 `power` 字段，电源区显示默认空闲 1 秒/确认 60 秒。导入参数文件后请手动重设电源模式、空闲和确认时间，再保存。直接保留原 `/data` 升级可沿用已保存配置；不要将配置导入视为完整电源设置迁移。
