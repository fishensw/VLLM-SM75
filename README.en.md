# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

SM75 compatibility and kernel optimizations, kept in sync with upstream [vLLM](https://github.com/vllm-project/vllm).

vLLM-SM75 v0.1.3 is based on vLLM 0.28.0 and integrates MTP, DFlash2 and auto-sleep support.

## v0.1.3 Update Summary

- FlashQLA-SM75 GDN prefill, Triton decode, FlashInfer 0.6.18, Marlin FP8 and FP8 KV.
- SM75 CUDA Graph adaptation and fused GDN metadata preparation.
- Native MTP verification optimizations and an MTP5 preset.
- DFlash2 SM75 numerical compatibility, AWQ dtype and TP4 adaptations.
- Reduced allocation pressure when loading a draft after an FP8 target.
- ModelScope support and persistent model/compiler caches.
- Idle auto-sleep with transparent wake-up; the launch script defaults to deep sleep after 30 minutes.
- Adds idle auto-sleep: after an idle timeout the engine automatically
  offloads its weights to free GPU memory and wakes automatically on the next
  request — keeping weights in pinned CPU memory, discarding and reloading
  them from the checkpoint, or exiting the engine process entirely (deep
  sleep, transparently cold-restarted on the next request). See "Idle
  auto-sleep" below.

## Performance

Environment: vLLM-SM75 v0.1.2, four Tesla T10 16 GiB GPUs, TP4, PCIe 3.0 x8, CUDA 12.9, PyTorch 2.13.0 and FlashInfer 0.6.18.

Qwen3.8-27B FP8 / W4A16-AWQ: repeated instruction text followed by a Python tool generation request; one request at a time, 1024 output tokens, temperature=0, thinking disabled, no prefix cache hits. Input counts include the chat template. Each row is one acceptance-test measurement.

| Configuration | Input tokens | TTFT (s) | Decode (tok/s) | Total time (s) |
| :------------ | -----------: | -------: | -------------: | -------------: |
| FP8 ordinary  |         1173 |    1.823 |          37.51 |         29.094 |
| FP8 ordinary  |        36885 |   35.434 |          35.62 |         64.152 |
| FP8 MTP5      |         1173 |    3.035 |          83.76 |         15.248 |
| FP8 MTP5      |        36885 |   36.400 |          76.41 |         49.789 |
| FP8 DFlash2   |         1173 |    1.154 |          98.12 |         11.581 |
| FP8 DFlash2   |        36885 |   35.002 |         113.81 |         43.990 |
| AWQ ordinary  |         1173 |    1.903 |          52.16 |         21.515 |
| AWQ ordinary  |        36885 |   34.337 |          48.55 |         55.410 |
| AWQ MTP5      |         1173 |    2.856 |          97.27 |         13.373 |
| AWQ MTP5      |        36885 |   34.988 |          99.80 |         45.239 |
| AWQ DFlash2   |         1173 |    1.156 |         122.51 |          9.507 |
| AWQ DFlash2   |        36885 |   34.106 |         140.84 |         41.369 |

TTFT measures time to the first text output; decode excludes prefill. Memory budgets and launch parameters are listed in the [test configurations](docs/validation/v0.1.2.md#测试配置).

Historical FP8 DFlash7 repetitive-text stress test, seq4/batch8192, 1024 output tokens: 32K median **147.70 tok/s**, highest run **151.28 tok/s**, near-100% acceptance. See the [release notes](docs/releases/v0.1.2.zh-CN.md) for those test conditions.

Real-world performance varies with hardware, models, request content and configuration; results below these test figures are normal.

## Quick reproduction

[Build and inference validation](docs/validation/v0.1.2.md) passed for 7 configurations and 14 requests.

### Persistent compilation caches

The launch script persists `/root/.cache/vllm` and `/root/.cache/flashinfer` to avoid repeated compilation on subsequent starts.

| Compilation phase | Time |
| :---------------- | ---: |
| Without cache reuse (historical record) | About 4–5 minutes |
| Cache hit (current FP8 MTP5 run) | **3.70 seconds** |

This run took approximately **202 seconds** from the first startup log to API startup. These records come from separate runs; cache acceleration requires matching compiled artifacts.

### 1. Clone

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
```

### 2. Build

Requires Linux x86_64, Docker with BuildKit, Git and Bash. Inference additionally requires an NVIDIA driver and NVIDIA Container Toolkit.

```bash
bash docker/build.sh
```

Uses the digest-pinned official `vllm/vllm-openai:v0.28.0-cu129` image, installs the adaptations and compiles the SM75 extension to produce `vllm-sm75:v0.1.3`.

### 3. Run

```bash
export VLLM_API_KEY='replace-with-your-api-key'
# Replace with an actual absolute host path for persistent model/compiler caches.
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75-cache
VARIANT=base FORMAT=fp8 bash docker/run.sh
```

The default FP8 model is `Qwen/Qwen3.8-27B-FP8`, resolved through ModelScope. The script sets the API key, port, listening address and cache mounts. Stop the previous GPU service before selecting another mode; the script does not stop existing services.

The firefly prefill is on by default in the image (`VLLM_FIREFLY=1`; applies to int4 weights + fp16/bf16 activations and W8A8-FP8 large-M prefill); to run a pure W4A16 baseline set `-e VLLM_FIREFLY=0`. `docker/run.sh` does not yet forward this variable, so run `docker run -e VLLM_FIREFLY=0 …` manually or append `--env VLLM_FIREFLY=0` to the script's `docker run` block.

```bash
VARIANT=mtp FORMAT=fp8 bash docker/run.sh
# Download the matching draft into your chosen host model directory first.
export MODEL_ROOT=/path/to/downloaded-models
DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2 VARIANT=dflash2 FORMAT=fp8 \
  bash docker/run.sh
```

MTP requires compatible MTP weights. Use `incoai/Qwen3.8-27B-DFlash2` as the matching draft.
For `philbert440/Qwen3.8-27B-W4A16-AWQ`, download the model and set `MODEL=/models/Qwen3.8-27B-W4A16-AWQ FORMAT=awq`.

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models \
  --header "Authorization: Bearer $VLLM_API_KEY"
```

See [build and launch details](docker/BUILD-v0.1.3.md).

## firefly (int4/fp8 weights -> int8 prefill acceleration)

To enable this, set `VLLM_FIREFLY=1` (off by default) per the launch section above;
`MIN_M`/`MODE` use defaults. Related environment variables:

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
  --auto-sleep-idle-timeout 30 \
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

vLLM changes retain Apache-2.0. FlashQLA-SM75 retains its MIT license and [source attribution](vllm/third_party/flash_qla_sm75/SOURCE.md).
