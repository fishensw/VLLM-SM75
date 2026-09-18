# vLLM-SM75

![VLLM-SM75](ultra/source/console/branding/icon-64.png)

[简体中文](README.md) | [English](README.en.md)

面向 Turing SM75 的 vLLM 0.29.0 适配，保留 FlashQLA GDN prefill、Triton decode、FlashInfer 0.6.18、Marlin FP8、Firefly AWQ prefill、FP8 all-reduce、MTP/DFlash2、CPU KV offload 和持久编译缓存。

当前源码版本为 **v0.1.5**，源码与更新说明见 [GitHub Release](https://github.com/fishensw/VLLM-SM75/releases/tag/v0.1.5)。Docker 镜像需按下文自行构建，尚未提供公开镜像。标准版提供推理 API 和 `/monitor`；ultra 在相同推理基础上增加 Web 控制台、工作台、随机 token 登录、模型/配置管理及实时监控。

QQ 交流群：**878924874**

## v0.1.5 更新说明

本次新增运行中投机解码开关，默认采用常驻 P-State 省电，并推出 ultra 版本。主要更新如下；前三项标准版与 ultra 均包含。

- **运行中切换投机解码**：通过 `/monitor` 查看状态、开关草稿计算，无需重启；需预先配置草稿及 SM75Scheduler，关闭不卸载草稿或释放其显存。
- **P-State 常驻省电**：新增 `POWER_MODE=pstate`，空闲进入 P8，有请求或负载时恢复驱动自动性能状态，模型保持加载；需匹配的 NVAPI 驱动库。
- **监控缓存修复**：启用监控看板不再使推理编译缓存失效，保留已有缓存兼容性。
- **新增 ultra 版本**：在标准版推理服务上增加 Web 管理、快速会话和工作台。

[版本更新与选择](docs/releases/v0.1.5.zh-CN.md) · [版本验证记录](docs/validation/v0.1.5.md)

### ultra 新增功能

ultra 包含标准版推理服务，并增加以下功能：

- 新增 Web 管理、快速会话和 DSH 工作台的统一使用入口。
- 首次随机 token、持久 Web 会话，修复导航及刷新过程的登录界面闪跳。
- 快速会话/工作台共用即时显示的页头：运行与 P-State、GPU 温度/功耗/核心/显存、P/D、KV、命中；性能监控页避免重复。
- 完善配置草稿、进程清理和 DSH 降权；账户登录、客户端网段白名单仍为后续功能。

[ultra 发布说明](docs/releases/v0.1.5-ultra.zh-CN.md) · [登录、升级与回退](ultra/README.md)

## 版本选择

**只需要推理 API，或已有聊天客户端/管理平台，选标准版；希望在浏览器中管理模型、直接聊天和使用工作台，选 ultra。**

| 项目 | 标准版 v0.1.5 | v0.1.5-ultra |
|---|---|---|
| 定位 | 推理服务 | 推理服务 + 一体化 Web 管理与工作台 |
| SM75 推理、FP8/AWQ、MTP/DFlash2、CPU KV | 支持 | 相同推理基座 |
| 运行中投机开关、P-State / 可选 sleep | 支持 | 支持 |
| 配置方式 | 启动脚本、环境变量和命令行参数 | Web 模型库与运行配置 |
| 监控 | 引擎 `/monitor` | 引擎监控 + 管理页实时状态 |
| Web token 登录、会话恢复 | 不包含 | 包含 |
| 快速会话、DSH 工作台 | 不包含 | 包含 |
| 默认端口 | 推理 API / monitor：8000 | 管理 / 聊天 / 工作台：1615；推理 API：8000 |
| 本地构建镜像 | `vllm-sm75:v0.1.5` | `vllm-sm75:v0.1.5-ultra` |

ultra 增加管理和交互功能，不代表推理内核更快。两版按需要选择一个部署即可；标准版不需要安装 ultra 才能提供推理 API。

## 性能参考

以下为 **v0.1.4 历史实测**，供参考，不是 v0.1.5/ultra 新实测。PCIe 3.0 ×8，4× Tesla T10 16 GiB、TP4、DFlash2 draft7，单轮 512 输出、无前缀命中。

补充参考：据群友实测反馈，PCIe 3.0 ×16 下 Prefill 极限约 **1800 tok/s**。

| 配置 | 输入 | Prefill tok/s | Decode tok/s | TTFT 秒 |
|---|---:|---:|---:|---:|
| FP8 DFlash2 | 32K | 1205.85 | 179.12 | 27.17 |
| FP8 DFlash2 | 128K | 978.19 | 166.39 | 133.99 |
| AWQ DFlash2 | 32K | 1433.67 | 215.57 | 22.86 |
| AWQ DFlash2 | 128K | 1124.88 | 198.40 | 116.52 |

重复文本的高接受率不代表日常聊天速度；FP8/AWQ 参数不同。详见 [历史 FP8](docs/validation/v0.1.4.md)、[历史 AWQ](docs/validation/v0.1.4-awq.md)、[本版验证记录](docs/validation/v0.1.5.md)。

## 快速复现

唯一版本源是 `docker/VERSION`，公共入口只有 `docker/build.sh` 和 `docker/run.sh`。Linux x86_64、Docker、Bash；运行另需 NVIDIA 驱动及 Container Toolkit。

```bash
# 标准版；在独立构建机执行完整构建
bash docker/build.sh
# ultra 复用刚构建的标准版
EDITION=ultra bash docker/build.sh

# 标准版推理，模型目录和缓存目录替换为实际绝对路径
export VLLM_API_KEY='replace-with-your-api-key'
export VLLM_SM75_CACHE_ROOT=/path/to/cache
VARIANT=base FORMAT=fp8 bash docker/run.sh

# ultra 首次安装：使用独立持久数据目录，不自动切换现有生产服务
ULTRA_DATA_ROOT=/path/to/ultra MODEL_ROOT=/path/to/models \
  EDITION=ultra bash docker/run.sh
```

镜像分别为 `vllm-sm75:v$(cat docker/VERSION)`、`vllm-sm75:v$(cat docker/VERSION)-ultra`，是本地构建标签，不表示已公开发布。`IMAGE` 可指定独立候选标签。ultra 默认控制台端口 1615、推理端口 8000；首次启动在持久目录生成 token，获取与登录见 [ultra 指南](ultra/README.md)。

**ultra 首次登录**：在 Docker 宿主机终端执行以下命令，复制输出的一整行 token，打开 `http://<局域网IP>:1615` 粘贴登录：

```bash
docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

自定义容器名时替换 `vllm-sm75-ultra`。token 位于容器 `/data/key`，对应宿主机 `<ULTRA_DATA_ROOT>/console/key`；它与模型 API key 分开。Unraid 容器 Console 内的命令及更多说明见 [首次登录](ultra/README.md#首次登录自动密钥在哪里怎么看)。

- [统一构建/运行指南](docker/BUILD.md)：FP8/AWQ、普通/MTP/DFlash2、缓存、电源、fast 与 UI 迭代。
- [目录约定](docker/README.md)：版本不进入脚本文件名，历史实现通过 Git 获取。
- [当前发布状态与门禁](docs/releases/v0.1.5.md)：源码、线上热修复、完整构建候选分别说明。
- [机器可读版本清单](docs/releases/v0.1.5-release-manifest.json)。

## Firefly

沿用 AWQ INT4 prefill 和 FP8 all-reduce；FP8 线性计算仍走 Marlin。默认及关闭开关见 [构建运行指南](docker/BUILD.md)，AWQ 历史配置见 [推荐配置](docs/recommended-awq-dflash2.md)。本轮不新增 AWQ 性能提升声明。

## `/monitor` 监控看板

标准引擎访问 `http://<主机>:8000/monitor`；ultra 另提供 1615 管理界面的性能监控。ultra Web 登录与引擎端点的访问控制范围不同，不能把 Web 登录当作整个推理端口的保护。

## 空闲自动休眠

默认使用常驻 P-State：空闲 P8，负载恢复高态 16（交还驱动自动控制）。自动休眠 exit 保留为可选配置，选择 `POWER_MODE=sleep` 后默认空闲 30 分钟退出引擎；下一请求重建并复用编译缓存，不保存对话 KV 或内存快照。模式条件、CPU 内存预算和回退见 [休眠与缓存](docs/sleep-and-cache.md)、[构建运行](docker/BUILD.md)。

## 历史与许可

[v0.1.4 发布说明](docs/releases/v0.1.4.zh-CN.md) · [v0.1.3](docs/releases/v0.1.3.zh-CN.md) · [v0.1.2](docs/releases/v0.1.2.zh-CN.md) · [开发验收历史](docs/releases/v0.1.5-ultra-rc.md)

保留 [LICENSE](LICENSE) 及第三方许可。图标以 vLLM 为主体、闪电表示本分支加速适配，不表示上游官方背书。
