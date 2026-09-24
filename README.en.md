# vLLM-SM75 v0.1.6

[中文 / full deployment guide](README.md) · [Release notes](docs/releases/v0.1.6.md) · [Configuration templates](docs/configuration-v0.1.6.md)

## 1. Overview

A vLLM adaptation for NVIDIA Turing / SM75 GPUs, including Tesla T4 and T10. It focuses on Qwen hybrid GDN models, FP8/AWQ weights, FP8 KV, and DFlash2. GDN uses FlashQLA-SM75 prefill and Triton decode.

QQ community group: **878924874**

## 2. Updates

- Adapt to vLLM 0.30.0 while retaining existing SM75, GDN, FP8 KV, Firefly, sleep and monitoring support.
- Backport quantized DFlash context KV, draft packed-module mapping and CUTLASS capability fixes; avoid the observed SM75 FP16/tensor-channel FP8 Humming overflow path through Marlin fallback.
- Add independent YaRN, per-GPU KV and total CPU KV settings, including a 1M configuration target.
- Optional LMCache 0.5.5 cu129 with mandatory packed-page compatibility fixes: CPU L1 and filesystem L2. The project-pinned extension is not installed or enabled by default.
- Ultra pins Harness 0.1.7-alpha.2, Node 22.23.2 and pnpm 11.7.0.
- Ultra adds context and cache controls, with a choice of default CPU KV or LMCache.

## 3. Standard vs Ultra

| Feature | Standard | Ultra |
| --- | --- | --- |
| OpenAI-compatible API / SM75 inference | Included | Same engine |
| Base, MTP, DFlash2 | CLI configuration | UI and CLI configuration |
| Engine monitor and speculative toggle | Port 8000 `/monitor` | Integrated workbench |
| P-State / sleep | Environment variables | Per-profile controls |
| Model, cache, launch settings | CLI | Templates, editor, import/export |
| YaRN and GPU/CPU KV | Explicit arguments | Separate validated settings |
| LMCache | Optional image extension; same-container service | Optional installation and managed service |
| Chat, images and tool workspace | External client | Quick chat and Harness |
| Web accounts / usage history | External management | Built in |
| Ports | API 8000 | Management 1615, API 8000 |

The UI itself is not an inference optimization. Do not start both editions on the same GPUs at once.

## 4. Performance reference

Full-image measurements: 4×T10 16 GiB, PCIe 3.0 ×8, TP4, identical FP8 target and DFlash2 draft7, 8 GiB total CPU KV and about 3.06 GiB GPU KV per card. Fixed input, three-position retrieval plus sustained prose, 512 output tokens, three measured repetitions after warmup. All 27 performance samples completed retrieval and output with zero cache hits and zero preemptions.

| Input | v0.1.5 Ultra Prefill / Decode | v0.1.6 standard Prefill / Decode | v0.1.6 Ultra Prefill / Decode |
| --- | ---: | ---: | ---: |
| 8192 | 1271.71 / 62.74 | 1282.27 / 65.78 | 1281.34 / 60.49 |
| 32768 | 1203.77 / 58.84 | 1209.02 / 64.46 | 1207.62 / 63.47 |
| 131072 | 976.08 / 59.85 | 977.80 / 53.01 | 978.08 / 58.73 |

All rates are tok/s. Prefill is input/TTFT; decode excludes the first streamed token batch. See [conditions, acceptance rates and evidence](docs/validation/v0.1.6-full-images.md). Standard 128K decode is 11.43% below the old baseline; Ultra stays within the 5% throughput tolerance at every length. **Continuation token hashes differ across versions and between standard/Ultra: strict equivalence acceptance failed.** These results do not establish lossless replacement or isolate console overhead.

Both standard and Ultra passed two 261632-input + 512-output boundary requests with no preemption; all four output token hashes match. Repeated prompts recomputed prefill without CPU KV readback. Historical [FP8](docs/validation/v0.1.4.md)/[AWQ](docs/validation/v0.1.4-awq.md) measurements use different workloads and timing; do not mix them with this table.

Context means input plus output. YaRN changes position configuration; CPU KV and LMCache reuse prefixes. Neither adds GPU attention capacity for an active request. Only completed requests with correct content count as tested context lengths.

## 5. Quick build

Use Linux x86_64, Git, Bash and Docker/BuildKit, with access to Docker Hub, PyPI, the PyTorch wheel index, FlashInfer and npm. The host does not need a separate PyTorch or CUDA compiler installation. Serving requires an appropriate NVIDIA driver and NVIDIA Container Toolkit; NVAPI is additionally required for P-State control.

The standard build starts from the fixed official vLLM 0.30.0 cu129 image, compiles SM75 FlashQLA, installs adaptations and runs checks. Ultra adds locked Node/Harness dependencies and the console. Allow roughly 200 GiB free build/artifact storage; the isolated validation worker uses 8 GiB RAM, 2 CPUs and MAX_JOBS=1. Serving has a separate memory budget.

The local formal Ultra image is tagged `vllm-sm75:v0.1.6-ultra`. To rebuild from release source on an isolated Linux host, verify the version first.

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
# Select the reviewed release source; docker/VERSION must read 0.1.6.
cat docker/VERSION
MAX_JOBS=1 bash docker/build.sh
EDITION=ultra bash docker/build.sh
# Optional instead of the last command:
# EDITION=ultra INSTALL_LMCACHE=1 bash docker/build.sh
```

Outputs: `vllm-sm75:v0.1.6` and `vllm-sm75:v0.1.6-ultra`. The tested archives use `local/vllm-sm75:v0.1.6-rc1` / `local/vllm-sm75:v0.1.6-ultra-rc1`; set `IMAGE` when starting those artifacts. See [archive checks and import](docs/validation/v0.1.6-full-images.md#构建与产物). Full builds through the unisolated production Unraid path remain blocked; see [isolated build requirements](docker/BUILD.md#隔离完整构建).

Standard, four-GPU startup after downloading the complete model:

```bash
export MODEL_ROOT=/srv/models
export MODEL=/models/Qwen3.8-27B-FP8
export VLLM_SM75_CACHE_ROOT=/srv/vllm-sm75/cache
export VLLM_SM75_MODEL_CACHE_ROOT=/srv/vllm-sm75/downloads
export VLLM_API_KEY='replace-with-your-api-key'
export PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1  # Use the actual host path.
VARIANT=base MAX_MODEL_LEN=32768 CPU_KV_GIB=2 POWER_MODE=pstate bash docker/run.sh
```

Ultra, using new persistent directories for a first installation:

```bash
ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra MODEL_ROOT=/srv/models \
  PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1 \
  EDITION=ultra bash docker/run.sh
docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

Open `http://<LAN-host>:1615`, log in with the generated Web token, register a model under `/models`, configure a profile and start it. The Web token and model API key are separate. Model startup is not automatic. Preserve `/data`, `/data/cache`, `/dsh/home` and `/dsh/workspace` during an upgrade; see [Ultra guide](ultra/README.md).

[Configuration templates](docs/configuration-v0.1.6.md) cover DFlash2, long context, YaRN, CPU/GPU KV, LMCache and the Ultra panel. P-State keeps weights/KV resident: low state 8, high value 16 restores driver control. CLI idle timeout is in seconds (default 1800); sleep timeout is in minutes (default 30). Ultra stores power settings per profile. P-State needs matching host NVAPI/NVML libraries and only one controller per GPU; it does not free VRAM.

[License](LICENSE) · [Sleep/cache](docs/sleep-and-cache.md) · [LMCache](docs/lmcache.md)
