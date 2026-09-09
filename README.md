# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

持续同步官方 [vLLM](https://github.com/vllm-project/vllm)，完善 SM75 兼容支持与内核优化。

vLLM-SM75 v0.1.3 基于 vLLM 0.28.0，集成 MTP、DFlash2 和自动休眠适配。

## v0.1.3 更新简要

- 新增空闲自动休眠与透明唤醒，启动脚本默认空闲 30 分钟后进入深度休眠。
- 修正空闲计时起点，从请求完成并进入 idle 状态后开始计时。
- 同版本修复 sleep 参数及引擎重建导致的编译缓存失效，DFlash2 草稿与候选选择器也复用缓存；镜像仍为 `vllm-sm75:v0.1.3`。

## 功能要点

- FlashQLA-SM75 GDN prefill、Triton decode、FlashInfer 0.6.18、Marlin FP8 和 FP8 KV。
- SM75 CUDA Graph、GDN 状态准备融合及原生 MTP5 验证路径。
- DFlash2 的 SM75 数值兼容、AWQ 数据类型和 TP4 处理。
- ModelScope、模型缓存与 vLLM/FlashInfer 编译缓存持久化。
- 统一镜像支持普通推理、MTP5 和 DFlash2，通过启动参数选择模式。
- 空闲自动睡眠支持 CPU、reload 和 exit；exit 模式释放引擎进程、CUDA context、worker 和显存，下一请求透明冷启动。

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

启动脚本将 vLLM 和 FlashInfer 编译缓存保存在宿主机，包含 DFlash2 草稿模型及候选选择器的编译产物。重建容器时保留缓存挂载，可复用匹配缓存，避免重复编译。

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
# 替换为实际绝对路径：编译缓存与模型下载目录分开。
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75/cache
export VLLM_SM75_MODEL_CACHE_ROOT=/path/to/model-cache

VARIANT=base FORMAT=fp8 bash docker/run.sh
```

FP8 默认通过 ModelScope 加载 `Qwen/Qwen3.8-27B-FP8`。脚本配置监听地址、端口、API key 和持久化挂载。使用相同 GPU 的模式按需互斥启动，脚本不会停止现有服务。

firefly prefill 镜像默认开（`VLLM_FIREFLY=1`，对 int4 权重 + fp16/bf16 激活、W8A8-FP8 大 M prefill 生效），要纯 W4A16 基线运行时用 `-e VLLM_FIREFLY=0` 关。`docker/run.sh` 暂未透传该变量，需手动 `docker run -e VLLM_FIREFLY=0 …` 或在脚本 `docker run` 段追加 `--env VLLM_FIREFLY=0`。

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

已完成本地 GPU 验证的配置、可复制的完整命令与实测效果见[FP8 DFlash2 推荐配置](docs/recommended-fp8-dflash2.md)。

## firefly（大 M prefill 加速）

v0.1.3 镜像默认开（`VLLM_FIREFLY=1`），要纯 W4A16 基线运行时用 `-e VLLM_FIREFLY=0` 关；非镜像直接用 env 时 envs.py 默认关（设 `VLLM_FIREFLY=1` 开）。仅加速大 M prefill，decode 和权重加载不变。

适用：int4 权重（W4A16/AWQ）及 W8A8-FP8 权重。

| 验证配置 | prefill 加速 |
| --- | --- |
| 2× T10, TP2, 27B W4A16/AWQ | 端到端验证通过 |
| 0.6B FP8 | 1.24× |

## 空闲自动休眠

请求完成并进入空闲后开始计时，达到设定时间自动休眠；新推理请求到达时自动恢复，API 服务保持在线。主要用途是降低长时间闲置时的显存占用和 GPU 功耗。

**日常省电、内存有限或使用 DFlash2，推荐 `exit`。** 启动脚本默认空闲 30 分钟后退出引擎及 GPU worker，下一请求自动重建；直接调用 `vllm serve` 则默认关闭自动休眠。

### 模式怎么选

| 模式 | 休眠和唤醒方式 | 必要条件 | 效果与限制 |
| --- | --- | --- | --- |
| **`exit`（推荐）** | 退出引擎和 worker；从已有磁盘模型文件重建，并复用匹配编译缓存 | 主模型、草稿模型及配置文件持续可读；保留缓存挂载；不需要 `--enable-sleep-mode` | 释放本引擎的 CUDA 上下文，本地已测四卡 P8；首个请求需等待完整重建 |
| `cpu` | 权重备份到 pinned CPU 内存，唤醒复制回 GPU | `--enable-sleep-mode`；额外 RAM 足够容纳实际权重备份（含草稿），另留服务内存 | 减少从磁盘重载权重的工作；保留进程和 CUDA 上下文，不保证 P8 |
| `reload` | 丢弃 GPU 权重，唤醒从 checkpoint 重载主模型 | `--enable-sleep-mode`；可读且支持重载的 checkpoint；**当前不要用于 DFlash2**，草稿不随主模型一起重载 | 不保留整模型权重备份，但进程、缓冲区等仍占内存；不保证 P8 |

exit/reload **不把运行时内存快照写入磁盘**，恢复来源是已有模型文件。编译缓存保存编译产物，不保存对话 KV；重启或 exit 唤醒后，长对话可能仍需重新处理输入。

内存与磁盘预算：cpu 需额外预留权重备份空间，不能简单按压缩模型文件大小计算；本地约 31 GiB 内存主机不采用整模型 cpu 备份。脚本原有 **8 GiB CPU KV offload** 是另一项内存开销，选择 exit/reload 不会取消它。磁盘保留完整主模型、草稿和编译缓存即可，无需单独准备休眠快照文件；文件页缓存也会使用可回收主机内存。

### 怎么配置

保留原推理参数，在 `vllm serve` 命令末尾添加：

```bash
--auto-sleep-idle-timeout 30 --auto-sleep-offload-target exit
```

超时单位是**分钟**：`1` = 60 秒测试，`30` = 日常 30 分钟，`0` = 关闭。使用仓库脚本时，在前文模型和路径设置的基础上任选对应配置：

```bash
# 60 秒测试
AUTO_SLEEP_IDLE_TIMEOUT=1 AUTO_SLEEP_OFFLOAD_TARGET=exit bash docker/run.sh
# 日常 30 分钟
AUTO_SLEEP_IDLE_TIMEOUT=30 AUTO_SLEEP_OFFLOAD_TARGET=exit bash docker/run.sh
# 关闭自动休眠
AUTO_SLEEP_IDLE_TIMEOUT=0 bash docker/run.sh
```

原有 `VARIANT`、`FORMAT`、模型等设置继续保留。已存在的同名容器需要按新参数重建，脚本不会自动替换它；重建时保留模型与编译缓存挂载。

| 参数 | 作用 / 默认值 |
| --- | --- |
| `--auto-sleep-idle-timeout` | 空闲分钟数；直接 CLI 默认 `0`，脚本变量 `AUTO_SLEEP_IDLE_TIMEOUT` 默认 `30` |
| `--auto-sleep-offload-target` | `exit` / `cpu` / `reload`；直接 CLI 默认 `cpu`，脚本变量 `AUTO_SLEEP_OFFLOAD_TARGET` 默认 `exit` |
| `--enable-sleep-mode` | cpu/reload 必需；脚本选择这两种模式时自动添加，exit 不需要 |
| `--auto-sleep-reload-path` | reload 的容器内 checkpoint 路径，默认启动模型路径；脚本变量 `AUTO_SLEEP_RELOAD_PATH` |
| `--auto-sleep-page-cache-keep-interval` | reload 文件页预热间隔，默认 `600` 秒；脚本变量 `AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL`。`0` 关闭睡眠时及后台预热，唤醒前仍提示预热一次 |

exit 只在退出前提示预热主模型文件页，没有后台预热进程。预热是 OS 提示，不能保证唤醒必定命中内存中的文件页。客户端及反向代理超时应覆盖完整唤醒时间。

### 已验证的推荐配置与效果

**FP8 DFlash2 + 30 分钟 exit**：4 × Tesla T10 16 GiB、TP4、约 31 GiB 主机 RAM，CPU 使用 ondemand。主模型 `Qwen/Qwen3.8-27B-FP8`，草稿 `incoai/Qwen3.8-27B-DFlash2`。

推理参数保持：**draft7、Graph `[8]`、seq4、batch8192、utilization 0.92、max-model-len 262144、每卡 KV 3288334336 bytes、FP8 e4m3 KV、8 GiB CPU KV offload**。这些显存预算针对该四卡环境；完整命令见[推荐配置](docs/recommended-fp8-dflash2.md)。

| 项目 | 本地实测结果 |
| --- | --- |
| 60 秒自动休眠 | API 在线，四卡连续三次采样均 P8；显存从 13663 降至 **3 MiB/卡** |
| 待机功耗 | 驻留单次快照约 **38–43 W/卡**；休眠三次样本约 **10–15.3 W/卡** |
| 自动唤醒 | 两个并发短请求均成功，完整请求耗时约 **135.7 秒** |
| 缓存复用 | 已有缓存启动、exit 唤醒、改为 30 分钟后启动，均 **12 次 AOT 命中、零重新编译**；包含主模型、草稿和候选选择器 |
| 缓存加载阶段 | 主模型 / 草稿 / 候选选择器约 **3.81 / 0.76 / 0.06 秒**，不等于完整唤醒耗时 |
| 正式 30 分钟配置 | 已启动并通过健康检查和推理；本轮未额外等待完整 30 分钟休眠周期 |

验收时先用 60 秒：完成一次推理后等待空闲超时及退出清理，确认 `/health` 在线，用下面命令观察 P-state、显存和功耗，再发请求确认能恢复，最后改回 30 分钟。

```bash
nvidia-smi --query-gpu=index,pstate,memory.used,power.draw --format=csv
```

P8 还取决于其他 GPU 进程、硬件和驱动，显存不要求绝对归零。当前 exit 验证环境为单 API server、DP=1、TP4；上述结果不代表其他模式、所有模型或 262K 长上下文都完成了本轮验收。CPU/reload 本轮只有状态机与参数测试，没有 GPU 唤醒性能保证。

## License

vLLM 修改遵循上游 Apache-2.0 License。FlashQLA-SM75 保留原始 MIT License 与[来源说明](vllm/third_party/flash_qla_sm75/SOURCE.md)。
