# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

项目目标是持续跟进官方
[vLLM](https://github.com/vllm-project/vllm)，完善其对 SM75 的兼容支持，
并优化相关内核。

当前 vLLM-SM75 v0.1.0 基于官方 vLLM `v0.28.0`，已在 4 x Tesla T10
16 GiB、CUDA 12.9、TP4 环境中使用 Qwen3.8 27B FP8 完成验证。

## 功能与改进

- 新增 `flashqla_sm75` GDN prefill backend。
- 集成 image-owned FlashQLA-SM75 CUDA 扩展，只编译 `sm_75`，默认禁止
  runtime JIT，不依赖宿主机 `.so` 挂载。
- FlashInfer 从官方基线的 0.6.16.post3 升级到 0.6.18，并恢复 SM75
  full-attention 能力选择。
- GDN prefill 与 decode 独立控制；最终配置使用 FlashQLA-SM75 prefill 和
  Triton decode。
- 不支持的 GPU、dtype 或 GDN head dimension 自动回退到 Triton/FLA。
- 保留 Marlin FP8、FP8 E4M3 KV cache、prefix cache、async scheduling、
  `FULL_AND_PIECEWISE` CUDA Graph 和 CPU KV offload。
- FlashInfer sampler 在 SM75 上关闭，attention 仍使用 FlashInfer，sampling
  回退到 vLLM 原生实现。
- 新增空闲自动睡眠（auto-sleep）：空闲超时后自动卸载权重释放显存，
  新请求到达自动唤醒（权重可备份到 CPU 内存、丢弃后从 checkpoint 重载，
  或直接退出引擎进程进入深度睡眠、下一请求透明冷启动），调用方无需任何
  额外接口。详见下文「空闲自动睡眠」。

FlashQLA 源码来自
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)，固定提交
`187b932dbd11940f0bcf52fb3675dd47fd69f313`。来源和许可证保留在
`vllm/third_party/flash_qla_sm75/`。

## 验证环境

| 组件 | 版本或配置 |
| --- | --- |
| vLLM-SM75 | v0.1.0 |
| 上游 vLLM | `v0.28.0` / `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| GPU | 4 x Tesla T10 16 GiB / SM75 |
| CUDA | 12.9 |
| PyTorch | 2.13.0 |
| FlashInfer | 0.6.18，无 `flashinfer-jit-cache` |
| 验证模型 | Qwen3.8 27B FP8 |
| Tensor parallel | TP4 / PYNCCL |
| GDN | FlashQLA-SM75 prefill + Triton decode |

## 实测结果

测试保持模型、TP、attention、KV、sampling 和服务资源一致，仅比较 GDN
prefill 路线。

| 指标 | vLLM v0.28.0 生产基线 | vLLM-SM75 v0.1.0 | 变化 |
| --- | ---: | ---: | ---: |
| Cold TTFT | 1.9562 s | 1.7082 s | **提升 12.68%** |
| Prefix-cached TTFT | 0.5064 s | 0.4378 s | **提升 13.55%** |
| Decode | 39.6289 token/s | 39.6306 token/s | 基本持平 |
| Cached tokens | 1,568 | 1,568 | 不变 |

确定性输出、streaming、tool calling、reasoning parser、多模态、prefix cache、
FP8 KV 和 8 GiB CPU KV offload 均通过验证。GDN 配置的 GPU KV 容量为 506,209
tokens，并观测到 821,297,152 bytes GPU 到 CPU KV 迁移。

## 快速复现

### 1. 克隆仓库

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
```

### 2. 构建镜像

准备兼容 vLLM 0.28.0、CUDA 12.9、PyTorch 2.13.0 和 SM75 的基础镜像：

```bash
export BASE_IMAGE='your-registry.example/vllm-openai:v0.28.0-cu129-sm75'

docker build \
  --file docker/Dockerfile.vllm-sm75-v0.1.0 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg BASE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --tag vllm-sm75-v0.1.0 \
  .
```

Dockerfile 会编译 SM75 扩展、检查 `cuobjdump` 架构、验证 `runtime` 版本，并在
构建阶段运行必要的 backend selector tests。

### 3. 启动服务

```bash
export VLLM_API_KEY='replace-with-your-api-key'
docker volume create vllm-hf-cache

docker run --detach --rm \
  --name vllm-sm75 \
  --gpus all \
  --shm-size 16g \
  --ulimit nofile=1048576:1048576 \
  --publish 8000:8000 \
  --volume vllm-hf-cache:/root/.cache/huggingface \
  --env VLLM_GDN_DECODE_KERNEL=triton \
  --env FLASH_QLA_SM75_ALLOW_JIT=0 \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env VLLM_USE_NCCL_SYMM_MEM=0 \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env VLLM_FIREFLY=1 \
  --entrypoint vllm \
  vllm-sm75-v0.1.0 \
  serve Qwen/Qwen3.8-27B-FP8 \
  --served-model-name VLLM-Qwen3.8-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --gpu-memory-utilization 0.87 \
  --tensor-parallel-size 4 \
  --max-model-len auto \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --attention-config '{"backend":"FLASHINFER"}' \
  --gdn-prefill-backend flashqla_sm75 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype float16 \
  --hf-overrides '{"dtype":"float16"}' \
  --generation-config vllm \
  --enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'

docker logs --follow vllm-sm75
```

本次 31 GiB 系统内存验证机在启用 8 GiB CPU KV offload 时同时启用了
16 GiB 主机 swap；内存更小且没有 swap 的主机可能在 offload 预分配阶段触发
OOM。`--shm-size 16g` 不能与 `--ipc=host` 同时使用，否则容器会重新受宿主机
`/dev/shm` 容量限制。

日志显示服务就绪后按 `Ctrl+C` 退出日志跟踪，再验证 OpenAI 兼容接口：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  --header "Authorization: Bearer $VLLM_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"model":"VLLM-Qwen3.8-27B","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

以上参数对应本项目的 4 x Tesla T10、TP4、Qwen3.8 27B FP8 验证配置。更换
GPU 数量、模型或可用显存后，再相应调整 TP、模型长度、并发和显存利用率。

## firefly（int4/fp8 权重 → int8 prefill 加速）

上面的启动命令已默认启用（`--env VLLM_FIREFLY=1`，`MIN_M`/`MODE` 用默认值）。
相关环境变量：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `VLLM_FIREFLY` | `0`（关闭） | 设 `1` 启用 firefly prefill。对 int4 权重 + fp16/bf16 激活、W8A8-FP8（fp8 权重，大 M prefill）生效，其它组合无影响（加载与 decode 均不变） |
| `VLLM_FIREFLY_MIN_M` | `1024` | batch M 超过该值才走 firefly prefill；小 M（decode）保持 int4 Marlin。反量化是 M 无关的固定成本，M 越小占比越高，可按实测调高 |
| `VLLM_FIREFLY_MODE` | `hard` | `hard` = 不额外存干净 int4 副本，prefill 时从 marlin 布局现反回（省显存，27B 可装下）；`easy` = 加载时存一份干净 int4 副本（反量化更快，但多约 0.5 字节/参数显存，大模型可能 OOM） |

注意事项：

- 仅 prefill 大 M 受影响；decode 与权重加载与上游逐 bit 一致。
- 对称 int4（W4A16/GPTQ，zp=8）与非对称 int4（AWQ，per-group zero point）都支持，
  反量化统一 `w_deq = (q - zp) * s`。
- int8 GEMM 引入一道额外的权重 int8 量化折叠，prefill 输出与 int4 Marlin 有
  微小数值差异（cos_sim > 0.9999 量级），不影响通顺度。
- 需要镜像内 CUDA 工具链支持 `sm_75` 的 `torch.utils.cpp_extension` JIT；
  无 GPU 或编译失败时自动回退 PyTorch 反量化（正确但慢约 50ms/层），不影响
  服务启动。
- 已在 2 x Tesla T10（SM75）、TP2、Qwen3.8 27B W4A16/AWQ（int4）上完成端到端
  验证；W8A8-FP8（fp8 权重）也支持：大 M prefill 现反量化 fp8 → int8
  （0.6B-fp8 验证 e2e prefill 1.24×，输出与 baseline 一致），decode 与加载不变。

## 空闲自动睡眠（auto-sleep）

引擎空闲超过设定时间后自动卸载权重、释放 GPU 显存；新请求到达时自动唤醒
（或重建）并继续服务，调用方无需任何额外调用。默认关闭
（`--auto-sleep-idle-timeout 0`），显式传入超时时开启。

示例配置（空闲 5 分钟自动进入**深度睡眠**：整个引擎进程退出，显存、
CUDA context、worker 进程全部归零；下一个请求透明地冷启动重拉）：

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  ... \
  --auto-sleep-idle-timeout 5 \
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

vLLM 修改继续遵循上游 Apache-2.0 License。FlashQLA-SM75 文件保留其原始
MIT License 和来源说明。
