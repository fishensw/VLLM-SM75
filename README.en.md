# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

SM75 compatibility and kernel optimizations, kept in sync with upstream [vLLM](https://github.com/vllm-project/vllm).

vLLM-SM75 v0.1.4 is based on vLLM 0.29.0 and integrates MTP, DFlash2 and auto-sleep support.

## v0.1.4 Update Summary

- **Firefly prefill acceleration**: improves AWQ INT4 throughput for long inputs.
- **FP8 all-reduce optimization**: reduces multi-GPU communication overhead and improves prefill performance.
- **Auto-sleep compatibility**: retains low-power idle and automatic wake-up, with stronger concurrency handling.
- **New `/monitor` dashboard**: displays runtime status and performance metrics.
- **Compatibility with upstream vLLM 0.29.0**.

### Compatibility fixes

Adapts DFlash2 weight loading and KV layouts to the new APIs, corrects CPU KV group identification, adds DFlash input/sampling warmup, and completes Triton/extension cache mounts.

See [release notes](docs/releases/v0.1.4.zh-CN.md), [FP8 measurements](docs/validation/v0.1.4.md), [AWQ measurements](docs/validation/v0.1.4-awq.md) and [AWQ configuration](docs/recommended-awq-dflash2.md). The throughput improvement concerns prefill, not a general decode speedup.

## v0.1.3 Update Summary

- Adds idle auto-sleep with transparent wake-up; the launch script defaults to deep sleep after 30 minutes.
- Starts the idle timer when a request completes and the engine becomes idle.

## Features

- FlashQLA-SM75 GDN prefill, Triton decode, FlashInfer 0.6.18, Marlin FP8 and FP8 KV.
- SM75 CUDA Graph, fused GDN metadata preparation, and the native MTP5 verification path.
- DFlash2 SM75 numerical compatibility, AWQ dtype and TP4 handling.
- ModelScope support and persistent model, vLLM and FlashInfer compilation caches.
- One image supports ordinary inference, MTP5 and DFlash2, selected by launch parameters.
- Idle auto-sleep supports CPU, reload and exit; exit releases the engine process, CUDA context, workers and GPU memory, then transparently cold-starts on the next request.

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

The launcher persists vLLM, FlashInfer, Triton and PyTorch extension caches on the host, including matching DFlash2 draft and selector artifacts. Keep model and compilation directories separate and preserve mounts when recreating containers. First use and code/dependency/configuration changes can still require compilation; cache loading is only part of startup.

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

Uses the digest-pinned official `vllm/vllm-openai:v0.29.0-cu129` image, installs the adaptations and compiles the SM75 extension to produce `vllm-sm75:v0.1.4`.

### 3. Run

Pass the model and options directly after the image name in `docker run`, without another `serve`. Commands executed inside a container still use `vllm serve ...`. Remove a leading `serve` from older container argument lists when migrating. `/monitor` remains an additional page.



```bash
export VLLM_API_KEY='replace-with-your-api-key'
# Use actual absolute host paths; keep model downloads separate.
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75/cache
export VLLM_SM75_MODEL_CACHE_ROOT=/path/to/model-cache
VARIANT=base FORMAT=fp8 bash docker/run.sh
```

The default FP8 model is `Qwen/Qwen3.8-27B-FP8`, resolved through ModelScope. The script sets the API key, port, listening address and cache mounts. Stop the previous GPU service before selecting another mode; the script does not stop existing services.

Firefly controls are forwarded by `docker/run.sh`; see the defaults and disable switches above.

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

See [build and launch details](docker/BUILD-v0.1.4.md).

## Firefly

INT4 uses the large-prefill path. FP8 linear computation stays on Marlin; the FP8 optimization tested here is quantized all-reduce. Small messages fall back to NCCL, with a default 1 MiB threshold and automatic backend selection. AWQ throughput has been measured at 1–128K; model quality and the final integrated fixes still require validation.

## `/monitor` dashboard

Open `http://HOST:PORT/monitor`. The self-contained page polls same-origin `/metrics`; no CDN or separate monitoring deployment is required. Set the container environment variable `VLLM_MONITOR=0` and recreate the container to disable it (`docker run -e VLLM_MONITOR=0`). The launcher does not separately forward this host variable. The dashboard and metrics currently do not require the model API key.

## Idle auto-sleep

For low idle GPU power and DFlash2, use `exit`. The API stays online while the engine and workers exit; the next inference request transparently rebuilds them. The launcher defaults to 30 minutes and exit, whereas direct `vllm serve` defaults to disabled auto-sleep and target cpu. Add to your existing serve arguments:

```bash
--auto-sleep-idle-timeout 30 --auto-sleep-offload-target exit
```

The timeout is in **minutes**: use `1` for a 60-second test, `30` for daily use, and `0` to disable. Idle timing begins after requests finish.

| Mode | Required resources and flags | Behavior |
| --- | --- | --- |
| `exit` | Readable main/draft model files and persistent compilation caches; no `--enable-sleep-mode` required | Releases engine CUDA contexts; full rebuild is paid by the first request |
| `cpu` | Extra pinned host RAM for actual weight allocations, including draft weights; `--enable-sleep-mode` | Restores weights from RAM; keeps processes/contexts, so P8 is not guaranteed |
| `reload` | Readable checkpoint; `--enable-sleep-mode`; currently do not use with DFlash2 | Discards weights and reloads the main model; processes, buffers and other CPU allocations remain |

The launcher supplies `--enable-sleep-mode` for cpu/reload. Optional variables are `AUTO_SLEEP_RELOAD_PATH` (container path) and `AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL` (seconds, default 600). Setting the latter to 0 disables reload's sleep-entry and background file-page warm-up; the one-shot hint before wake-up remains. OS cache residency is not guaranteed. Exit issues a one-shot main-model hint before exiting.

Neither exit nor reload writes a runtime-memory snapshot to disk. Budget for normal loading memory and the launcher's existing 8 GiB CPU KV offload separately from pinned weight backups. Keep full main/draft weights and leave disk space for compilation artifacts. A migrated host path is safe only when the container-side path and matching artifacts remain available; model/code/TP/dtype changes can still invalidate caches.

FP8 DFlash2 exit was measured on T10 ×4 with the API online: P8 on all cards, 3 MiB per card, 9.97–15.31 W. Other GPU users and drivers can prevent P8. Validated topology is single API server, DP=1, TP4; cpu/reload GPU behavior and other topologies were not validated in this cache-fix run. Client/proxy timeouts must accommodate the full wake-up. Do not mix automatic sleep with development-only manual sleep/wake endpoints.

See the [complete guide and resource requirements (Chinese)](docs/sleep-and-cache.md). This draft targets `vllm-sm75:v0.1.4`; historical sleep measurements are not combined-image acceptance.

### Experimental disk snapshots

The existing local `disk` backend saves model allocations and restores them in place. KV contents are invalidated and CUDA contexts remain alive; P8 is not guaranteed. It requires a writable real-disk mount and snapshot capacity. It is not part of the recommended v0.1.4 configuration or this GPU acceptance round.

## Validation scope

FP8/AWQ measurements belong to the compatibility-patched PR image. The combined image passed its build, AWQ startup and a single-request check. CPU KV recovery and a full sleep/P8 cycle were not repeated in this round. Historical v0.1.3 sleep results do not validate v0.1.4. Full data and conditions are linked above.

## License

vLLM changes retain Apache-2.0. FlashQLA-SM75 retains its MIT license and [source attribution](vllm/third_party/flash_qla_sm75/SOURCE.md).
