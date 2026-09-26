# vLLM-SM75 v0.1.6

[English](README.en.md) · [发布说明](docs/releases/v0.1.6.md) · [参数模板](docs/configuration-v0.1.6.md) · [完整构建](docker/BUILD.md)

## 1. 简要说明

面向 NVIDIA Turing / SM75（Tesla T4、T10 等）的 vLLM 0.30.0 适配，保留 GDN、FP8/AWQ、FP8 KV、MTP/DFlash2、Firefly、CPU KV、自动休眠和监控等功能。标准版提供推理 API；Ultra 在相同推理核心上提供 Web 管理、快速会话和工作台。

QQ 交流群：**878924874**

## 2. 更新说明

本版主要更新如下；推理及缓存相关功能由标准版和 Ultra 共用。

- **引擎适配**：适配 vLLM 0.30.0，保留 SM75、GDN、FP8 KV、Firefly、自动休眠和监控等原有功能。
- **DFlash2 修复**：修复量化 context KV、草稿权重映射和 CUTLASS 能力判断；SM75 避开 Humming 提前溢出的路径，回退到 Marlin。
- **编译缓存隔离**：更新内部编译标识，避免复用不匹配的内核和权重布局缓存。
- **长上下文配置**：新增 YaRN 1M、每卡 GPU KV 和整组 CPU KV 的独立设置，保留模型原生位置编码字段。
- **LMCache 分层缓存**：可选 CPU L1、磁盘 L2 与重启复用，默认不安装本项目固定扩展、不启用。
- **Ultra 工作台**：升级到 Harness 0.1.7-alpha.2，固定 Node 22.23.2 和 pnpm 11.7.0。
- **Ultra 面板**：新增上下文与缓存配置区，支持选择默认 CPU KV 或 LMCache。

## 3. 标准版与 Ultra 功能对比

| 功能 | 标准版 | Ultra |
| --- | --- | --- |
| OpenAI 兼容 API、SM75 推理核心 | 有 | 相同核心 |
| 普通推理 / MTP / DFlash2 | 启动参数 | 面板配置及启动参数 |
| 引擎监控、投机开关 | `:8000/monitor`，需按引擎鉴权 | 工作台集成，确认实际生效状态 |
| P-State 常驻省电 / 自动休眠 | 环境变量 | 每个运行配置的电源设置 |
| 模型、启动参数、编译缓存管理 | 命令行 | Web 管理、模板、导入导出 |
| YaRN、GPU KV、CPU KV | 显式参数 | 可视化配置并校验 |
| LMCache CPU＋磁盘 | 可选扩展层；同容器管理缓存服务 | 可选安装；面板管理进程与缓存目录 |
| 对话、图片附件、工具工作区 | 使用外部客户端 | 快速对话＋Harness 工作台 |
| 管理员登录、网段策略、历史用量 | 由外部管理系统提供 | 内置 |
| 默认端口 | 推理 8000 | 管理 1615；推理 8000 |

选标准版适合已有前端和运维流程；Ultra 适合在一个容器中管理模型与工作台。相同参数下不应把面板本身描述为推理加速。两版不要同时争用同一组 GPU。

## 4. 性能参考

本轮完整镜像实测：4×T10 16 GiB、PCIe 3.0 ×8、TP4、同一 FP8 目标和 DFlash2 draft7、CPU KV 8 GiB、每卡 GPU KV 约 3.06 GiB。固定输入、三位置检索＋长篇续写、输出 512 token，各项预热后三次中位数。全部 27 个性能样本完成检索和输出，零缓存命中、零抢占。

| 输入 Token | v0.1.5 Ultra Prefill / Decode（tok/s） | v0.1.6 标准版 Prefill / Decode（tok/s） | v0.1.6 Ultra Prefill / Decode（tok/s） |
| --- | ---: | ---: | ---: |
| 8192 | 1271.71 / 62.74 | 1282.27 / 65.78 | 1281.34 / 60.49 |
| 32768 | 1203.77 / 58.84 | 1209.02 / 64.46 | 1207.62 / 63.47 |
| 131072 | 976.08 / 59.85 | 977.80 / 53.01 | 978.08 / 58.73 |

## 5. 快速构建与启动

### 依赖与构建流程

- Linux x86_64、Bash、Git、Docker / BuildKit；构建机需访问 Docker Hub、PyPI、PyTorch wheel 索引、FlashInfer 和 npm。宿主不需要安装 PyTorch/CUDA 编译工具；隔离流程的源码打包另需 Python 3（仅标准库）。
- 运行机需兼容 Turing 的 NVIDIA 驱动与 NVIDIA Container Toolkit，先确认容器能访问 GPU。本轮宿主驱动为 610.43.02，这不是最低版本声明。
- 镜像内部固定 vLLM 0.30.0、Torch 2.13.0、CUDA 12.9、Transformers 5.15.1、FlashInfer 0.6.18；移除不适用于 SM75 的 FlashInfer JIT cache。完整清单见 [版本清单](docs/releases/v0.1.6-release-manifest.json)。
- 标准版：官方固定摘要基座 → 编译 SM75 FlashQLA → 安装适配 → 构建检查。Ultra：相同标准版 → 锁定 Harness/Node 依赖 → 控制台与插件。
- 全量构建需要保存数十 GiB 的基础层、缓存和产物，建议预留 **200 GiB** 空间；本轮隔离 worker 使用 **8 GiB RAM、2 CPU、MAX_JOBS=1**。这不是模型推理的内存预算。

本地正式 Ultra 镜像标签为 `vllm-sm75:v0.1.6-ultra`。以下是从发布源码完整构建的流程；在独立 Linux 构建机确认版本号后执行：

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
# 使用本次发布源码或已核对的提交；确认 docker/VERSION 为 0.1.6。
cat docker/VERSION
MAX_JOBS=1 bash docker/build.sh
EDITION=ultra bash docker/build.sh
# 若需要 LMCache，第二步改用：
# EDITION=ultra INSTALL_LMCACHE=1 bash docker/build.sh
```

得到 `vllm-sm75:v0.1.6` 和 `vllm-sm75:v0.1.6-ultra`。构建不会自动启动模型。若复用本轮验收归档，标签为 `local/vllm-sm75:v0.1.6-rc1` / `local/vllm-sm75:v0.1.6-ultra-rc1`，用 `IMAGE` 覆盖启动标签，见 [归档校验与导入](docs/validation/v0.1.6-full-images.md#构建与产物)。生产 Unraid 的未隔离完整导出入口仍被拦截；使用 [隔离构建流程](docker/BUILD.md#隔离完整构建) 保留宿主资源保护。

### 标准版：先验证基本推理

先将完整目标模型和匹配草稿下载到宿主模型目录；下列路径均需替换。启动脚本的默认模板按 **4 卡 T10、TP4** 编写。

```bash
export MODEL_ROOT=/srv/models
export MODEL=/models/Qwen3.8-27B-FP8
export VLLM_SM75_CACHE_ROOT=/srv/vllm-sm75/cache
export VLLM_SM75_MODEL_CACHE_ROOT=/srv/vllm-sm75/downloads
export VLLM_API_KEY='replace-with-your-api-key'
export PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1  # 换成实际驱动库路径
VARIANT=base POWER_MODE=pstate MAX_MODEL_LEN=32768 CPU_KV_GIB=2 bash docker/run.sh
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

首次启动会加载权重、编译和捕获 CUDA Graph；以 `/health` 和实际生成成功判断就绪。接入 DFlash2、256K、YaRN、CPU KV、LMCache 的可复制参数与效果见 [配置推荐模板](docs/configuration-v0.1.6.md)。

### Ultra：面板与首次登录

```bash
ULTRA_DATA_ROOT=/srv/vllm-sm75/ultra MODEL_ROOT=/srv/models \
  PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1 \
  EDITION=ultra bash docker/run.sh
# 在宿主终端读取首次自动生成的 Web 登录 token：
docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

打开 `http://<局域网IP>:1615` 登录，在面板登记 `/models` 下模型，创建运行配置，再启动推理。Web token 位于 `<ULTRA_DATA_ROOT>/console/key`，与模型 API key 分开。默认只启动管理服务。

**P-State 必备：** 匹配宿主驱动的 NVAPI `libnvidia-api.so.1`，以及由 NVIDIA Toolkit `utility` 提供的 NVML。标准启动脚本默认空闲 1800 秒、确认 60 秒、低态 8、高态 **16（恢复驱动自动控制，不是硬件 P16）**。Ultra 未设置电源项时的默认空闲为 1 秒、确认 60 秒；每份配置独立保存；当前“导入配置”不会带入电源字段，会回到上述默认值，保存前请在电源区重新填写。保留原数据目录升级的既有配置不受此导入行为影响。一个 GPU 只使用一个 P-State 控制器；不要设为 0/0。完整参数和单位见 [电源模板](docs/configuration-v0.1.6.md#电源模式与单位)。

升级 Ultra 保留原 `/data`、`/data/cache`、`/dsh/home`、`/dsh/workspace`；先用独立目录验收，再按 [升级与回退](ultra/README.md#首次使用与升级) 切换。首次安装命令不能用来覆盖现有数据。

---

[历史发布](docs/releases/v0.1.5-ultra.zh-CN.md) · [休眠与缓存](docs/sleep-and-cache.md) · [LMCache 详细说明](docs/lmcache.md) · [LICENSE](LICENSE)
