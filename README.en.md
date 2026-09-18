# vLLM-SM75

![VLLM-SM75](ultra/source/console/branding/icon-64.png)

[简体中文](README.md) | [English](README.en.md)

Turing SM75 adaptation of vLLM 0.29.0, retaining FlashQLA GDN prefill, Triton decode, FlashInfer 0.6.18, Marlin FP8, Firefly AWQ prefill, FP8 all-reduce, MTP/DFlash2, CPU KV offload and persistent compilation caches.

The current source is **v0.1.5**; source and release notes are available in the [GitHub Release](https://github.com/fishensw/VLLM-SM75/releases/tag/v0.1.5). Build Docker images locally using the instructions below; public container images are not provided. Standard provides the inference API and `/monitor`. Ultra adds the Web console, workbench, random-token login, model/configuration management and monitoring on the same inference base.

QQ community group: **878924874**

## v0.1.5 updates

This update adds runtime speculative-decoding control, default resident P-State power management, and the ultra edition. The main updates are listed below; the first three apply to both editions.

- **Toggle speculative decoding at runtime** through `/monitor` without restarting the model. A draft model and SM75Scheduler must already be configured; disabling drafting does not unload the draft or release its VRAM.
- **Resident P-State power mode**: `POWER_MODE=pstate` enters P8 when idle and restores driver-managed performance on requests or load, keeping the model loaded. A matching NVAPI driver library is required.
- **Dashboard cache fix**: enabling monitoring no longer invalidates inference compilation caches.
- **New optional ultra edition**: adds Web administration, quick chat and a workbench to the standard inference service.

[Edition guide and release notes (Chinese)](docs/releases/v0.1.5.zh-CN.md) · [Version validation record (Chinese)](docs/validation/v0.1.5.md)

### Additional ultra features

Ultra includes the standard inference service and adds the following features:

- Integrated Web administration, quick chat and DSH workbench.
- Random initial token, persistent Web sessions, and fixes for login flashes during navigation and refresh.
- Shared, immediately rendered quick-chat/workbench header: engine/P-State, GPU temperature, power, core utilization, VRAM, P/D, KV and cache hits. The performance page avoids a duplicate metric bar.
- Configuration drafts, process cleanup and DSH privilege dropping. Account login and client-network allowlists remain future work.

[Ultra release notes (Chinese)](docs/releases/v0.1.5-ultra.zh-CN.md) · [Login, upgrade and rollback](ultra/README.md)

## Choose an edition

**Choose standard for an inference API or an existing chat client/management platform. Choose ultra to manage models, chat and use the workbench in a browser.**

| Item | Standard v0.1.5 | v0.1.5-ultra |
|---|---|---|
| Purpose | Inference service | Inference + integrated Web management and workbench |
| SM75 inference, FP8/AWQ, MTP/DFlash2, CPU KV | Supported | Same inference base |
| Runtime speculative toggle, P-State / optional sleep | Supported | Supported |
| Configuration | Scripts, environment variables and CLI arguments | Web model library and runtime configuration |
| Monitoring | Engine `/monitor` | Engine monitoring + live management status |
| Web token login and session recovery | Not included | Included |
| Quick chat and DSH workbench | Not included | Included |
| Default ports | API / monitor: 8000 | Management / chat / workbench: 1615; API: 8000 |
| Local image tag | `vllm-sm75:v0.1.5` | `vllm-sm75:v0.1.5-ultra` |

Ultra adds management and interaction, not a faster inference kernel. Choose one edition for your deployment; standard does not require ultra to serve the inference API.

## Performance reference

These are **historical v0.1.4 measurements**, not new v0.1.5/ultra results. Hardware: PCIe 3.0 ×8, 4× Tesla T10 16 GiB, TP4, DFlash2 draft7; one run per entry, 512 output tokens, no prefix hits.

Additional reference: community members report measured peak Prefill throughput of approximately **1800 tok/s** over PCIe 3.0 ×16.

| Configuration | Input | Prefill tok/s | Decode tok/s | TTFT seconds |
|---|---:|---:|---:|---:|
| FP8 DFlash2 | 32K | 1205.85 | 179.12 | 27.17 |
| FP8 DFlash2 | 128K | 978.19 | 166.39 | 133.99 |
| AWQ DFlash2 | 32K | 1433.67 | 215.57 | 22.86 |
| AWQ DFlash2 | 128K | 1124.88 | 198.40 | 116.52 |

High acceptance on repetitive text is not representative of everyday chat; FP8 and AWQ settings differ. See [historical FP8](docs/validation/v0.1.4.md), [historical AWQ](docs/validation/v0.1.4-awq.md), and [current validation](docs/validation/v0.1.5.md).

## Quick reproduction

`docker/VERSION` is the only version source. The public entry points are `docker/build.sh` and `docker/run.sh`. Build on Linux x86_64 with Docker and Bash; inference also requires NVIDIA drivers and Container Toolkit.

```bash
# Complete build on a separate build host
bash docker/build.sh
EDITION=ultra bash docker/build.sh

export VLLM_API_KEY='replace-with-your-api-key'
export VLLM_SM75_CACHE_ROOT=/path/to/cache
VARIANT=base FORMAT=fp8 bash docker/run.sh

# Fresh ultra installation; never replaces an existing production container
ULTRA_DATA_ROOT=/path/to/ultra MODEL_ROOT=/path/to/models \
  EDITION=ultra bash docker/run.sh
```

**First ultra login:** run this command in the Docker host terminal, copy the entire output line, and paste it into the login form at `http://<LAN-IP>:1615`:

```bash
docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

Replace `vllm-sm75-ultra` if you set a custom container name. The automatically generated Web token is stored at `/data/key` inside the container, mapped to `<ULTRA_DATA_ROOT>/console/key` on the host. It is separate from the engine API key and persists when the data directory is retained. In Unraid's container Console, run `node /opt/sm75-workbench/console/auth-cli.mjs show` without `docker exec`.

Local image tags are `vllm-sm75:v$(cat docker/VERSION)` and `vllm-sm75:v$(cat docker/VERSION)-ultra`. Override `IMAGE` for independent candidate tags. Ultra uses console port 1615 and engine port 8000. It generates its login token in persistent storage; see the [ultra guide](ultra/README.md).

- [Unified build/run guide](docker/BUILD.md): inference modes, caches, power settings, fast/UI iteration.
- [Directory contract](docker/README.md): unversioned script names; historical implementations remain in Git.
- [Current release status and gates](docs/releases/v0.1.5.md).
- [Machine-readable manifest](docs/releases/v0.1.5-release-manifest.json).

## Firefly

AWQ INT4 prefill and FP8 all-reduce are retained; FP8 linear computation still uses Marlin. Configuration and disable switches are in the [build/run guide](docker/BUILD.md); historical AWQ settings are in the [AWQ guide](docs/recommended-awq-dflash2.md). No new AWQ speedup is claimed here.

## `/monitor` dashboard

The engine dashboard is at `http://<host>:8000/monitor`; ultra additionally provides performance monitoring through its management UI on port 1615. Web login and engine endpoint access controls have different scopes; Web login does not protect the entire inference port.

## Automatic idle sleep

Resident P-State is the default: idle P8 and high-state 16 under load (driver-managed performance). Exit sleep remains optional via `POWER_MODE=sleep`; its default idle timeout is 30 minutes. Exit rebuilds the engine on the next request using compatible compilation caches, without saving conversation KV or runtime memory snapshots. See [sleep/cache requirements](docs/sleep-and-cache.md) and the [build/run guide](docker/BUILD.md).

## History and licensing

[v0.1.4 notes](docs/releases/v0.1.4.zh-CN.md) · [v0.1.3](docs/releases/v0.1.3.zh-CN.md) · [v0.1.2](docs/releases/v0.1.2.zh-CN.md) · [Development history](docs/releases/v0.1.5-ultra-rc.md)

[LICENSE](LICENSE) and third-party notices are retained. The vLLM-based mark with a lightning accent identifies this acceleration adaptation, not an upstream endorsement.
