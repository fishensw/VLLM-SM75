# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

持续同步官方 [vLLM](https://github.com/vllm-project/vllm)，完善 SM75 兼容支持与内核优化。

vLLM-SM75 v0.1.3 基于 vLLM 0.28.0，集成 MTP、DFlash2 和自动休眠适配。

## v0.1.3 更新简要

- 保留 FlashQLA-SM75 GDN prefill、Triton decode、FlashInfer 0.6.18、Marlin FP8 和 FP8 KV 支持。
- 适配 SM75 CUDA Graph，融合 GDN 状态准备，减少投机验证及调度开销。
- 优化原生 MTP 验证路径，提供 MTP5 配置。
- 完善 DFlash2 的 SM75 数值兼容、AWQ 数据类型及 TP4 处理。
- 修复 FP8 加载 draft 时的显存分配压力。
- 支持 ModelScope、模型与编译缓存持久化。
- 新增空闲自动休眠与透明唤醒，启动脚本默认 30 分钟后进入深度休眠。
- 新增空闲自动睡眠（auto-sleep）：空闲超时后自动卸载权重释放显存，
  新请求到达自动唤醒（权重可备份到 CPU 内存、丢弃后从 checkpoint 重载，
  或直接退出引擎进程进入深度睡眠、下一请求透明冷启动），调用方无需任何
  额外接口。详见下文「空闲自动睡眠」。

## 性能参考

测试环境：vLLM-SM75 v0.1.2，4 × Tesla T10 16 GiB、TP4、PCIe 3.0 ×8、CUDA 12.9、PyTorch 2.13.0、FlashInfer 0.6.18。

Qwen3.8-27B FP8 / W4A16-AWQ：重复说明文本并要求生成 Python 工具，单请求、1024 输出 tokens、temperature=0、关闭 thinking、无前缀缓存命中。输入长度包含聊天模板；以下为各配置、各长度的单次验收实测。

| 配置        | 输入 tokens | TTFT（秒） | decode（tok/s） | 总耗时（秒） |
| :---------- | ----------: | ---------: | --------------: | -----------: |
| FP8 普通    |        1173 |      1.823 |           37.51 |       29.094 |
| FP8 普通    |       36885 |     35.434 |           35.62 |       64.152 |
| FP8 MTP5    |        1173 |      3.035 |           83.76 |       15.248 |
| FP8 MTP5    |       36885 |     36.400 |           76.41 |       49.789 |
| FP8 DFlash2 |        1173 |      1.154 |           98.12 |       11.581 |
| FP8 DFlash2 |       36885 |     35.002 |          113.81 |       43.990 |
| AWQ 普通    |        1173 |      1.903 |           52.16 |       21.515 |
| AWQ 普通    |       36885 |     34.337 |           48.55 |       55.410 |
| AWQ MTP5    |        1173 |      2.856 |           97.27 |       13.373 |
| AWQ MTP5    |       36885 |     34.988 |           99.80 |       45.239 |
| AWQ DFlash2 |        1173 |      1.156 |          122.51 |        9.507 |
| AWQ DFlash2 |       36885 |     34.106 |          140.84 |       41.369 |

TTFT 为首个文本输出等待时间，decode 不含 prefill。各配置的显存预算与启动参数见[测试配置](docs/validation/v0.1.2.md#测试配置)。

历史 FP8 DFlash7 重复文本压力测试：seq4、batch8192、1024 输出 tokens，32K decode 中位数 **147.70 tok/s**，单轮最高 **151.28 tok/s**，接受率接近 100%；详细条件见[更新说明](docs/releases/v0.1.2.zh-CN.md)。

实际场景复杂，性能随硬件、模型、请求内容和配置变化，未达到测试数据是正常现象。

## 快速复现

v0.1.2 的基础推理验收见[验证记录](docs/validation/v0.1.2.md)；v0.1.3 新增自动休眠测试。

### 编译缓存持久化

启动脚本已持久化 `/root/.cache/vllm` 和 `/root/.cache/flashinfer`，避免后续启动重复编译。

| 编译阶段 | 耗时 |
| :------- | ---: |
| 未复用缓存（历史记录） | 约 4–5 分钟 |
| 命中缓存（本次 FP8 MTP5） | **3.70 秒** |

本次完整启动约 **202 秒**。以上为不同轮次记录；缓存加速适用于已有匹配编译缓存的启动。

### 1. 克隆

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
```

### 2. 构建

Linux x86_64，需安装 Docker、Git 和 Bash。启动模型另需 NVIDIA 驱动与 NVIDIA Container Toolkit。

```bash
bash docker/build.sh
```

基于固定 digest 的官方 `vllm/vllm-openai:v0.28.0-cu129` 镜像，安装全部适配并编译 SM75 扩展，生成 `vllm-sm75:v0.1.3`。

### 3. 启动

```bash
export VLLM_API_KEY='replace-with-your-api-key'
# 替换为实际绝对路径：持久化模型、vLLM 编译及 FlashInfer 缓存。
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75-cache

VARIANT=base FORMAT=fp8 bash docker/run.sh
```

FP8 默认通过 ModelScope 加载 `Qwen/Qwen3.8-27B-FP8`。脚本配置监听地址、端口、API key 和持久化挂载。使用相同 GPU 的模式按需互斥启动，脚本不会停止现有服务。

启用 firefly prefill：设 `VLLM_FIREFLY=1`（默认关闭，对 int4 权重 + fp16/bf16 激活、W8A8-FP8 大 M prefill 生效）。`docker/run.sh` 暂未透传该变量，需手动 `docker run --env VLLM_FIREFLY=1 …` 或在脚本 `docker run` 段追加 `--env VLLM_FIREFLY=1`。

```bash
# MTP5：使用同一个镜像
VARIANT=mtp FORMAT=fp8 bash docker/run.sh

# DFlash2：先将匹配 draft 下载到自选目录
export MODEL_ROOT=/path/to/downloaded-models
DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2 VARIANT=dflash2 FORMAT=fp8 \
  bash docker/run.sh
```

DFlash draft 使用 `incoai/Qwen3.8-27B-DFlash2`，完整模型文件放入上述挂载目录。
AWQ 使用 `philbert440/Qwen3.8-27B-W4A16-AWQ`，下载后设置 `MODEL=/models/Qwen3.8-27B-W4A16-AWQ FORMAT=awq`。其余配置见[构建与启动说明](docker/BUILD-v0.1.3.md)。

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models \
  --header "Authorization: Bearer $VLLM_API_KEY"
```

## firefly（大 M prefill 加速）

v0.1.3 镜像默认开（`VLLM_FIREFLY=1`），要纯 W4A16 基线运行时用 `-e VLLM_FIREFLY=0` 关；非镜像直接用 env 时 envs.py 默认关（设 `VLLM_FIREFLY=1` 开）。仅加速大 M prefill，decode 和权重加载不变。

适用：int4 权重（W4A16/AWQ）及 W8A8-FP8 权重。

| 验证配置 | prefill 加速 |
| --- | --- |
| 2× T10, TP2, 27B W4A16/AWQ | 端到端验证通过 |
| 0.6B FP8 | 1.24× |

## 空闲自动睡眠（auto-sleep）

引擎空闲超过设定时间后自动卸载权重、释放 GPU 显存；新请求到达时自动唤醒
（或重建）并继续服务，调用方无需任何额外调用。默认关闭
（`--auto-sleep-idle-timeout 0`），显式传入超时时开启。

示例配置（空闲 30 分钟自动进入**深度睡眠**：整个引擎进程退出，显存、
CUDA context、worker 进程全部归零；下一个请求透明地冷启动重拉）：

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  ... \
  --auto-sleep-idle-timeout 30 \
  --auto-sleep-offload-target exit
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--auto-sleep-idle-timeout` | `0`（关闭） | 空闲超过该分钟数触发自动睡眠；float，如 `0.2` = 12 秒（短窗口测试用） |
| `--auto-sleep-offload-target` | `cpu` | `cpu` = 权重备份到 CPU 内存（sleep level 1，唤醒约 1-2s，占约 30 GiB 主机内存，需 `--enable-sleep-mode`）；`reload` = 权重直接丢弃、唤醒时从 checkpoint 重载（sleep level 2，不占 CPU 内存，唤醒约 20-60s，需 `--enable-sleep-mode`）；`exit` = 退出整个引擎进程（深度睡眠，显存/CUDA context/worker 全归零，GPU 进 P8 待机，下一请求透明冷启动约 1-3 min，**无需** `--enable-sleep-mode`） |
| `--auto-sleep-reload-path` | 启动时的模型路径 | `reload` 模式唤醒使用的 checkpoint 路径 |
| `--auto-sleep-page-cache-keep-interval` | `600` | `reload` 模式下，睡眠期间每隔该秒数把 checkpoint 重新预热进 OS page cache，保证唤醒时的读盘走缓存而非冷读 NVMe；`0` = 关闭后台预热（睡眠/唤醒瞬间仍会各预热一次）。`exit` 模式在退出前预热一次，让冷启动读盘走缓存 |

注意事项：

- 唤醒耗时计入空闲后第一个请求的 TTFT：`reload` 模式需要从磁盘读
  checkpoint 并重跑量化 repack（约 20-60 秒）；`cpu` 模式约 1-2 秒；
  `exit` 模式是完整冷启动（重建进程 + 模型加载 + 量化 repack，约 1-3 分钟），
  换取睡眠期间 GPU 完全空闲。
- `exit` 模式（深度睡眠）退出整个引擎进程，睡眠期间显存、CUDA context、
  worker 进程全部归零，省电最彻底；代价是唤醒最慢。适合长时间空闲
  （如夜间）的场景。目前仅支持单 API server、DP=1 的部署拓扑。
- `reload` 模式要求模型 checkpoint 在磁盘上持续可读（即
  `vllm-hf-cache` 卷保持挂载）。
- `reload` 模式默认在睡眠期间把 checkpoint 预热进 OS page cache（每 600 秒
  一次，`--auto-sleep-page-cache-keep-interval` 可调、设 `0` 关闭）。这能让
  唤醒时的 `reload_weights` 读盘命中缓存而非冷读，NVMe 场景下可缩短唤醒
  耗时约 10-25 秒；对已在缓存中的页是零开销的空操作。
- CPU 内存有限的主机（如 31 GiB 验证机）推荐 `reload` 或 `exit`；`cpu` 模式
  需要额外约 30 GiB 主机内存存放 pinned 备份。
- 启用投机解码 drafter 时推荐 `cpu` 模式（drafter 权重不随主模型
  自动重载）。
- 手动 `POST /sleep` / `POST /wake_up` / `GET /is_sleeping` 位于 dev
  路由；auto 模式下手动调用的行为未定义。

## License

vLLM 修改遵循上游 Apache-2.0 License。FlashQLA-SM75 保留原始 MIT License 与[来源说明](vllm/third_party/flash_qla_sm75/SOURCE.md)。
