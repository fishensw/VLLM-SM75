# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

This project tracks upstream
[vLLM](https://github.com/vllm-project/vllm), improves its SM75 compatibility,
and optimizes the relevant kernels.

The current vLLM-SM75 v0.1.0 release is based on upstream vLLM `v0.28.0`.
It has been validated with Qwen3.8 27B FP8 on 4 x Tesla T10 16 GiB GPUs,
CUDA 12.9, and TP4.

## Features and improvements

- Adds the `flashqla_sm75` GDN prefill backend.
- Integrates a self-contained FlashQLA-SM75 CUDA extension that compiles only
  for `sm_75`, disables runtime JIT by default, and requires no host-mounted
  `.so` files.
- Upgrades FlashInfer from the upstream baseline of 0.6.16.post3 to 0.6.18 and
  restores SM75 full-attention capability selection.
- Controls GDN prefill and decode independently. The final configuration uses
  FlashQLA-SM75 for prefill and Triton for decode.
- Automatically falls back to Triton/FLA for unsupported GPUs, dtypes, or GDN
  head dimensions.
- Retains Marlin FP8, FP8 E4M3 KV cache, prefix caching, async scheduling,
  `FULL_AND_PIECEWISE` CUDA Graph, and CPU KV offload.
- Disables the FlashInfer sampler on SM75 while retaining FlashInfer attention;
  sampling falls back to the native vLLM implementation.
- Adds idle auto-sleep: after an idle timeout the engine automatically
  offloads its weights to free GPU memory and wakes automatically on the next
  request — keeping weights in pinned CPU memory, discarding and reloading
  them from the checkpoint, or exiting the engine process entirely (deep
  sleep, transparently cold-restarted on the next request). See "Idle
  auto-sleep" below.

The FlashQLA source is derived from
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) at commit
`187b932dbd11940f0bcf52fb3675dd47fd69f313`. Its provenance and license are
preserved under `vllm/third_party/flash_qla_sm75/`.

## Validated environment

| Component | Version or configuration |
| --- | --- |
| vLLM-SM75 | v0.1.0 |
| Upstream vLLM | `v0.28.0` / `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| GPU | 4 x Tesla T10 16 GiB / SM75 |
| CUDA | 12.9 |
| PyTorch | 2.13.0 |
| FlashInfer | 0.6.18, without `flashinfer-jit-cache` |
| Validated model | Qwen3.8 27B FP8 |
| Tensor parallel | TP4 / PYNCCL |
| GDN | FlashQLA-SM75 prefill + Triton decode |

## Benchmark results

The benchmark keeps the model, tensor parallelism, attention, KV cache,
sampling, and service resources unchanged, and compares only the GDN prefill
path.

| Metric | vLLM v0.28.0 production baseline | vLLM-SM75 v0.1.0 | Change |
| --- | ---: | ---: | ---: |
| Cold TTFT | 1.9562 s | 1.7082 s | **12.68% faster** |
| Prefix-cached TTFT | 0.5064 s | 0.4378 s | **13.55% faster** |
| Decode | 39.6289 token/s | 39.6306 token/s | Essentially unchanged |
| Cached tokens | 1,568 | 1,568 | Unchanged |

Deterministic output, streaming, tool calling, reasoning parser, multimodal
input, prefix cache, FP8 KV cache, and 8 GiB CPU KV offload all passed
validation. The GDN configuration provides a GPU KV capacity of 506,209
tokens, with an observed 821,297,152-byte GPU-to-CPU KV transfer.

## Quick reproduction

### 1. Clone the repository

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
```

### 2. Build the image

Prepare a base image compatible with vLLM 0.28.0, CUDA 12.9, PyTorch 2.13.0,
and SM75:

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

The Dockerfile compiles the SM75 extension, validates the target architecture
with `cuobjdump`, verifies runtime versions, and runs the required backend
selector tests during the build.

### 3. Start the service

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

The 31 GiB RAM validation host also used 16 GiB of host swap with the 8 GiB
CPU KV offload configuration. Smaller hosts without swap may be OOM-killed
during offload preallocation. Do not combine `--shm-size 16g` with
`--ipc=host`, because host IPC bypasses the container's private shm size.

After the log reports that the service is ready, press `Ctrl+C` to stop
following the log and verify the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  --header "Authorization: Bearer $VLLM_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"model":"VLLM-Qwen3.8-27B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

These parameters reproduce the validated 4 x Tesla T10, TP4, Qwen3.8 27B FP8
configuration. Adjust tensor parallelism, model length, concurrency, and GPU
memory utilization when changing the GPU count, model, or available memory.

## firefly (int4/fp8 weights -> int8 prefill acceleration)

The launch command above enables this by default (`--env VLLM_FIREFLY=1`,
`MIN_M`/`MODE` use defaults). Related environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_FIREFLY` | `0` (off) | Set to `1` to enable firefly prefill. Applies to int4 weights + fp16/bf16 activations and W8A8-FP8 (fp8 weights, large-M prefill); other combinations are unaffected (loading and decode remain unchanged) |
| `VLLM_FIREFLY_MIN_M` | `1024` | Firefly prefill kicks in only when batch M exceeds this value; small M (decode) keeps int4 Marlin. Dequantization is a fixed cost independent of M, so smaller M means a higher overhead ratio — tune up based on measurements |
| `VLLM_FIREFLY_MODE` | `hard` | `hard` = no clean int4 copy stored; prefill dequantizes on the fly from the marlin layout (saves memory, fits 27B); `easy` = stores a clean int4 copy at load time (faster dequantization, but adds ~0.5 byte/param of memory, may OOM on large models) |

Notes:

- Only large-M prefill is affected; decode and weight loading are bit-identical
  to upstream.
- Both symmetric int4 (W4A16/GPTQ, zp=8) and asymmetric int4 (AWQ, per-group
  zero point) are supported; dequantization is unified as `w_deq = (q - zp) * s`.
- The int8 GEMM introduces one extra weight int8 quantization folding step;
  prefill output has a tiny numerical difference from int4 Marlin
  (cos_sim > 0.9999 magnitude), no impact on readability.
- Requires the in-image CUDA toolchain to support `sm_75` via
  `torch.utils.cpp_extension` JIT; without a GPU or on compilation failure it
  automatically falls back to PyTorch dequantization (correct but ~50ms/layer
  slower), without affecting service startup.
- End-to-end validated on 2 x Tesla T10 (SM75), TP2, Qwen3.8 27B W4A16/AWQ
  (int4); W8A8-FP8 (fp8 weights) is also supported: large-M prefill dequantizes
  fp8 -> int8 (validated on 0.6B-fp8, 1.24x e2e prefill, outputs match baseline),
  decode and loading unchanged.

## Idle auto-sleep

After an idle timeout the engine automatically offloads its weights to free
GPU memory, and wakes (or rebuilds) automatically when a new request arrives;
callers need no extra API calls. Disabled by default
(`--auto-sleep-idle-timeout 0`); set a timeout to turn it on.

Example configuration (auto-sleep after 5 idle minutes into **deep sleep**:
the whole engine process exits — GPU memory, CUDA context, and worker
processes all go to zero; the next request transparently cold-restarts it):

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  ... \
  --auto-sleep-idle-timeout 5 \
  --auto-sleep-offload-target exit
```

| Flag | Default | Description |
| --- | --- | --- |
| `--auto-sleep-idle-timeout` | `0` (disabled) | Auto-sleep after this many idle minutes; float, e.g. `0.2` = 12 s for short test windows |
| `--auto-sleep-offload-target` | `cpu` | `cpu`: pinned CPU backup (sleep level 1, wake ~1-2 s, ~30 GiB host RAM, needs `--enable-sleep-mode`); `reload`: discard weights, reload from the checkpoint on wake (sleep level 2, no CPU memory, wake ~20-60 s, needs `--enable-sleep-mode`); `exit`: terminate the engine-core process entirely (deep sleep — GPU memory, CUDA context, and workers all go to zero; the GPU drops to its deepest idle state; transparently cold-restarted on the next request, ~1-3 min; does **not** need `--enable-sleep-mode`) |
| `--auto-sleep-reload-path` | startup model path | Checkpoint used to reload weights on wake in `reload` mode |
| `--auto-sleep-page-cache-keep-interval` | `600` | In `reload` mode, re-warm the checkpoint into the OS page cache every this many seconds while sleeping, so the wake-time disk read hits the cache instead of cold NVMe; `0` disables the background warm (a one-shot warm on sleep/wake still happens). In `exit` mode the checkpoint is warmed once just before exit so the cold start reads from cache |

Notes:

- Wake time is paid by the first request after idleness: `reload` mode reads
  the checkpoint from disk and re-runs the quantization repack (~20-60 s);
  `cpu` mode takes ~1-2 s; `exit` mode is a full cold start (process +
  model load + quantization repack, ~1-3 min) in exchange for the GPU being
  completely idle while asleep.
- `exit` mode (deep sleep) terminates the engine process — the most thorough
  power saving — at the cost of the slowest wake. It suits long idle periods
  (e.g. overnight). Currently supported only for the single-API-server,
  DP=1 topology.
- `reload` mode requires the model checkpoint to stay readable on disk (the
  `vllm-hf-cache` volume must remain mounted).
- `reload` mode warms the checkpoint into the OS page cache by default while
  sleeping (every 600 s; tune or disable with
  `--auto-sleep-page-cache-keep-interval`). This lets the wake-time
  `reload_weights` read hit the cache instead of cold NVMe, cutting wake time
  by roughly 10-25 s on NVMe; it is a no-op for pages already resident.
- On hosts with limited CPU RAM (e.g. the 31 GiB validation host), prefer
  `reload` or `exit`; `cpu` mode needs ~30 GiB of extra pinned CPU memory.
- With a speculative decoding drafter, prefer `cpu` mode (drafter weights
  are not reloaded together with the main model).
- Manual `POST /sleep` / `POST /wake_up` / `GET /is_sleeping` live on the
  dev endpoints; mixing them with auto mode is undefined.

## License

The vLLM modifications remain under the upstream Apache-2.0 License. The
FlashQLA-SM75 files retain their original MIT License and provenance notices.
