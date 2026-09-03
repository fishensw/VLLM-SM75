# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

This project tracks upstream
[vLLM](https://github.com/vllm-project/vllm), improves its SM75 compatibility,
and optimizes the relevant kernels.

The current vLLM-SM75 v0.1.1 release is based on upstream vLLM `v0.28.0`.
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
- Resolves model IDs through ModelScope by default and supports persistent
  model, vLLM compilation, and FlashInfer JIT caches across container rebuilds.

The FlashQLA source is derived from
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) at commit
`187b932dbd11940f0bcf52fb3675dd47fd69f313`. Its provenance and license are
preserved under `vllm/third_party/flash_qla_sm75/`.

## Validated environment

| Component | Version or configuration |
| --- | --- |
| vLLM-SM75 | v0.1.1 |
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

| Metric | vLLM v0.28.0 production baseline | vLLM-SM75 v0.1.1 | Change |
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
  --file docker/Dockerfile.vllm-sm75-v0.1.1 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg BASE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --tag vllm-sm75-v0.1.1 \
  .
```

The Dockerfile compiles the SM75 extension, validates the target architecture
with `cuobjdump`, verifies runtime versions, and runs the required backend
selector tests during the build.

### 3. Start the service

```bash
export VLLM_API_KEY='replace-with-your-api-key'

# Persistent cache root. It can be any absolute host path with enough space.
export VLLM_SM75_CACHE_ROOT="${VLLM_SM75_CACHE_ROOT:-$HOME/.cache/vllm-sm75}"

mkdir -p \
  "$VLLM_SM75_CACHE_ROOT/modelscope" \
  "$VLLM_SM75_CACHE_ROOT/vllm" \
  "$VLLM_SM75_CACHE_ROOT/flashinfer"

docker run --detach --rm \
  --name vllm-sm75 \
  --gpus all \
  --shm-size 16g \
  --ulimit nofile=1048576:1048576 \
  --publish 8000:8000 \
  --volume "$VLLM_SM75_CACHE_ROOT/modelscope:/root/.cache/modelscope:rw" \
  --volume "$VLLM_SM75_CACHE_ROOT/vllm:/root/.cache/vllm:rw" \
  --volume "$VLLM_SM75_CACHE_ROOT/flashinfer:/root/.cache/flashinfer:rw" \
  --env VLLM_USE_MODELSCOPE=true \
  --env MODELSCOPE_CACHE=/root/.cache/modelscope/hub \
  --env VLLM_GDN_DECODE_KERNEL=triton \
  --env FLASH_QLA_SM75_ALLOW_JIT=0 \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env VLLM_USE_NCCL_SYMM_MEM=0 \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --entrypoint vllm \
  vllm-sm75-v0.1.1 \
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

The three mounts persist downloaded ModelScope models, vLLM
`torch.compile`/AOT artifacts, and FlashInfer JIT kernels respectively. Their
contents survive container deletion and recreation, avoiding repeated model
downloads and compilation. If a vLLM, PyTorch, CUDA, or FlashInfer upgrade
causes a cache incompatibility, remove the corresponding `vllm` or
`flashinfer` subdirectory and regenerate it; the model cache can remain.

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

## License

The vLLM modifications remain under the upstream Apache-2.0 License. The
FlashQLA-SM75 files retain their original MIT License and provenance notices.
