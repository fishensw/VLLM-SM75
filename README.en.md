# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

SM75 compatibility and kernel optimizations, kept in sync with upstream [vLLM](https://github.com/vllm-project/vllm).

vLLM-SM75 v0.1.2 is based on vLLM 0.28.0 and integrates MTP and DFlash2 support.

## v0.1.2 Update Summary

- FlashQLA-SM75 GDN prefill, Triton decode, FlashInfer 0.6.18, Marlin FP8 and FP8 KV.
- SM75 CUDA Graph adaptation and fused GDN metadata preparation.
- Native MTP verification optimizations and an MTP5 preset.
- DFlash2 SM75 numerical compatibility, AWQ dtype and TP4 adaptations.
- Reduced allocation pressure when loading a draft after an FP8 target.
- ModelScope support and persistent model/compiler caches.

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
bash docker/build-v0.1.2.sh
```

Uses the digest-pinned official `vllm/vllm-openai:v0.28.0-cu129` image, installs the adaptations and compiles the SM75 extension to produce `vllm-sm75:v0.1.2`.

### 3. Run

```bash
export VLLM_API_KEY='replace-with-your-api-key'
# Replace with an actual absolute host path for persistent model/compiler caches.
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75-cache
VARIANT=base FORMAT=fp8 bash docker/run-v0.1.2.sh
```

The default FP8 model is `Qwen/Qwen3.8-27B-FP8`, resolved through ModelScope. The script sets the API key, port, listening address and cache mounts. Stop the previous GPU service before selecting another mode; the script does not stop existing services.

```bash
VARIANT=mtp FORMAT=fp8 bash docker/run-v0.1.2.sh
# Download the matching draft into your chosen host model directory first.
export MODEL_ROOT=/path/to/downloaded-models
DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2 VARIANT=dflash2 FORMAT=fp8 \
  bash docker/run-v0.1.2.sh
```

MTP requires compatible MTP weights. Use `incoai/Qwen3.8-27B-DFlash2` as the matching draft.
For `philbert440/Qwen3.8-27B-W4A16-AWQ`, download the model and set `MODEL=/models/Qwen3.8-27B-W4A16-AWQ FORMAT=awq`.

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models \
  --header "Authorization: Bearer $VLLM_API_KEY"
```

See [build and launch details](docker/BUILD-v0.1.2.md).

## License

vLLM changes retain Apache-2.0. FlashQLA-SM75 retains its MIT license and [source attribution](vllm/third_party/flash_qla_sm75/SOURCE.md).
