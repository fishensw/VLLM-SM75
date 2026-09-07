# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

持续同步官方 [vLLM](https://github.com/vllm-project/vllm)，完善 SM75 兼容支持与内核优化。

vLLM-SM75 v0.1.2 基于 vLLM 0.28.0，集成 MTP 和 DFlash2 适配。

## v0.1.2 更新简要

- 保留 FlashQLA-SM75 GDN prefill、Triton decode、FlashInfer 0.6.18、Marlin FP8 和 FP8 KV 支持。
- 适配 SM75 CUDA Graph，融合 GDN 状态准备，减少投机验证及调度开销。
- 优化原生 MTP 验证路径，提供 MTP5 配置。
- 完善 DFlash2 的 SM75 数值兼容、AWQ 数据类型及 TP4 处理。
- 修复 FP8 加载 draft 时的显存分配压力。
- 支持 ModelScope、模型与编译缓存持久化。

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

已通过[构建与运行验收](docs/validation/v0.1.2.md)：7 项配置、14 条请求。

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
bash docker/build-v0.1.2.sh
```

基于固定 digest 的官方 `vllm/vllm-openai:v0.28.0-cu129` 镜像，安装全部适配并编译 SM75 扩展，生成 `vllm-sm75:v0.1.2`。

### 3. 启动

```bash
export VLLM_API_KEY='replace-with-your-api-key'
# 替换为实际绝对路径：持久化模型、vLLM 编译及 FlashInfer 缓存。
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75-cache

VARIANT=base FORMAT=fp8 bash docker/run-v0.1.2.sh
```

FP8 默认通过 ModelScope 加载 `Qwen/Qwen3.8-27B-FP8`。脚本配置监听地址、端口、API key 和持久化挂载。使用相同 GPU 的模式按需互斥启动，脚本不会停止现有服务。

```bash
# MTP5：使用同一个镜像
VARIANT=mtp FORMAT=fp8 bash docker/run-v0.1.2.sh

# DFlash2：先将匹配 draft 下载到自选目录
export MODEL_ROOT=/path/to/downloaded-models
DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2 VARIANT=dflash2 FORMAT=fp8 \
  bash docker/run-v0.1.2.sh
```

DFlash draft 使用 `incoai/Qwen3.8-27B-DFlash2`，完整模型文件放入上述挂载目录。
AWQ 使用 `philbert440/Qwen3.8-27B-W4A16-AWQ`，下载后设置 `MODEL=/models/Qwen3.8-27B-W4A16-AWQ FORMAT=awq`。其余配置见[构建与启动说明](docker/BUILD-v0.1.2.md)。

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models \
  --header "Authorization: Bearer $VLLM_API_KEY"
```

## License

vLLM 修改遵循上游 Apache-2.0 License。FlashQLA-SM75 保留原始 MIT License 与[来源说明](vllm/third_party/flash_qla_sm75/SOURCE.md)。
