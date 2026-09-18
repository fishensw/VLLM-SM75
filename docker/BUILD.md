# 构建与启动

> 完整构建和临时停止相关模型容器均允许，前提是不拖垮宿主 Docker 服务、不影响无关容器，并能恢复原功能。此前未隔离的完整层导出引发故障，`docker/build.sh` 暂时拦截该 Unraid 路径；已验证的隔离构建过程见 [完整构建记录](../docs/validation/2026-09-18-isolated-full-build.md)；通用入口仍不允许生产 Unraid 的未隔离完整导出。完整构建使用独立构建机，UI 迭代使用下面的资源受限增量模式。不能只停模型后原样重试事故命令。

版本号统一来自 `docker/VERSION`（构建与启动脚本自动读取）；目录约定见 `docker/README.md`。

## 1. 准备

使用 Linux x86_64、Docker、Git、Bash；启动需要 NVIDIA 驱动及 NVIDIA Container Toolkit。四卡示例按 TP4 配置。构建下载官方镜像和公开依赖，仅编译 SM75 扩展，默认 `MAX_JOBS=1`。

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
```

## 2. 构建统一镜像

```bash
bash docker/build.sh
```

基础环境直接使用官方 `vllm/vllm-openai:v0.29.0-cu129`，固定 amd64 digest `sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2`。Dockerfile 安装 FlashInfer 0.6.18、移除不适用的 JIT cache 包，再安装本仓库适配、编译 FlashQLA 并运行构建检查。

产物：`vllm-sm75:v$(cat docker/VERSION)`，包含普通推理、MTP、DFlash2 和自动休眠支持，由启动参数选择模式。

### ultra 和开发模式

```bash
# 同一入口构建 ultra；先完成标准版构建
EDITION=ultra bash docker/build.sh
# 可选：明确指定已审计的标准版镜像与独立输出标签
VLLM_IMAGE=my-standard-candidate IMAGE=my-ultra-candidate EDITION=ultra bash docker/build.sh

# 开发基座，文件名及 tag 版本由同一版本源管理
BUILD_MODE=fast bash docker/build.sh
# 容器内使用挂载到 /work 的源码迭代；只用于无推理任务的开发容器
# bash /opt/vllm-sm75/fast_compile.sh

# ultra 静态/控制台增量：要求已有经过审计的 ultra 运行时镜像
EDITION=ultra BUILD_MODE=ui RUNTIME_IMAGE=my-audited-ultra IMAGE=my-ui-candidate bash docker/build.sh
```

UI 模式使用无网络、1 GiB 内存、1 CPU 的轻量覆盖构建，不证明完整源码可复现。完整构建默认不自动启动模型；它与部署是两个动作。`IMAGE` 不应指向正在使用的标签。fast 仅为迭代基座，不能作为完整标准版镜像直接启动服务。

## 3. 启动

```bash
export VLLM_API_KEY='replace-with-your-api-key'
# 替换成宿主机实际绝对路径；编译缓存与模型下载分开。
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75/cache
export VLLM_SM75_MODEL_CACHE_ROOT=/path/to/model-cache
VARIANT=base FORMAT=fp8 bash docker/run.sh
```

MTP5 使用 `VARIANT=mtp`，要求模型具有匹配 MTP 权重。普通和 MTP 使用自动 KV；FP8 使用 seq4/batch8192，AWQ 使用 seq8/batch16384，utilization 均为0.87、max-model-len=auto。

AWQ 或 DFlash 需要先下载模型及匹配 draft，放在自选目录，再只读挂载：

```bash
export MODEL_ROOT=/path/to/downloaded-models
mkdir -p "$MODEL_ROOT"
# 通过镜像自带的 CLI 下载，无需在宿主机安装 Python 环境。
docker run --rm --volume "$MODEL_ROOT:/models" --entrypoint modelscope \
vllm-sm75:v$(cat docker/VERSION) download --model incoai/Qwen3.8-27B-DFlash2 \
  --local_dir /models/Qwen3.8-27B-DFlash2
docker run --rm --volume "$MODEL_ROOT:/models" --entrypoint hf \
  vllm-sm75:v$(cat docker/VERSION) download philbert440/Qwen3.8-27B-W4A16-AWQ \
  --local-dir /models/Qwen3.8-27B-W4A16-AWQ

# 普通 AWQ：目录中放置已完整下载的 philbert440/Qwen3.8-27B-W4A16-AWQ。
MODEL=/models/Qwen3.8-27B-W4A16-AWQ VARIANT=base FORMAT=awq bash docker/run.sh
# FP8 DFlash：目录中放置已完整下载的 incoai/Qwen3.8-27B-DFlash2。
DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2 VARIANT=dflash2 FORMAT=fp8 bash docker/run.sh
```

以上是互斥启动示例；先停止已运行的同 GPU 服务，再选择另一种。脚本不会停止或删除现有容器。首次运行前需将所有 `/path/to/...` 改为实际路径，模型目录应包含配置、tokenizer 和完整权重。

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models \
  --header "Authorization: Bearer $VLLM_API_KEY"
```

测试必须包含新缓存启动、模型实际请求、SM75 扩展和投机路径日志核对。构建成功与模型推理通过分别记录。

## 省电模式：sleep / pstate（POWER_MODE）

### Linux 驱动库准备

在 **Docker 宿主机**安装适用于 GPU 的 NVIDIA 专有驱动，并配置 NVIDIA Container Toolkit。先确认宿主机 `nvidia-smi` 正常。P-State 需要以下两项驱动库，版本应与宿主驱动匹配；镜像内的管理程序不能替代宿主驱动。

| 驱动库 | 用途 | 容器内提供方式 |
|---|---|---|
| `libnvidia-api.so.1`（NVAPI） | P-State 控制 | 本项目从宿主只读挂载，可通过 `PSTATE_NVAPI_LIB` 指定 |
| `libnvidia-ml.so.1`（NVML） | GPU 状态、负载等查询 | NVIDIA Container Toolkit 的 `utility` 能力注入 |

软件包名称随发行版、驱动分支变化：Arch Linux 通常由 `nvidia-utils` 提供；Debian 的 NVAPI 包为 `libnvidia-api1` 或 `libnvidia-tesla-api1`，按已安装的驱动分支选择，NVML 包需另外核对。Ubuntu 等衍生版可能使用带驱动版本号的包名，不要混装其他分支的驱动库。来源见 [nvidia-pstated Linux 依赖](https://github.com/sasha0552/nvidia-pstated#prerequirements)。

Debian / Ubuntu 可先搜索可用包，再安装与当前驱动匹配的包：

```bash
apt search libnvidia-api
apt search libnvidia-ml
# 按文件名定位软件包（apt search 不负责检索包内文件）
sudo apt install apt-file
sudo apt-file update
apt-file search -x '/libnvidia-(api|ml)\.so\.1$'
# 确认包名和驱动分支后安装，例如 Debian 对应分支二选一：
# sudo apt install libnvidia-api1
# sudo apt install libnvidia-tesla-api1
```

Arch Linux 使用与当前驱动匹配的 `nvidia-utils` 包；Unraid 使用宿主 NVIDIA 驱动插件提供的匹配库，不执行上述 apt 命令。若驱动包未提供 NVAPI，需从同版本 NVIDIA 驱动发行包获取该库。

在宿主机检查库路径：

```bash
nvidia-smi
ldconfig -p | grep -E 'libnvidia-(api|ml)\.so\.1'
# 若 NVAPI 没有出现在链接器缓存，检查常见路径：
ls -l /usr/lib64/libnvidia-api.so.1 \
  /usr/lib/x86_64-linux-gnu/libnvidia-api.so.1 /usr/lib/libnvidia-api.so.1
```

`ls` 中部分路径不存在是正常的，取实际存在且可读的绝对路径设置 `PSTATE_NVAPI_LIB`。标准版自动检查上述三个路径；ultra 必须显式指定。两者均挂载到容器 `/usr/local/nvidia/lib64/libnvidia-api.so.1`。

NVML 通常不需要手动挂载。自定义 Docker / Compose 配置若限制了 `NVIDIA_DRIVER_CAPABILITIES`，应包含 `compute,utility`；其中 `utility` 用于 `nvidia-smi` 和 NVML，详见 [NVIDIA 容器驱动能力说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)。该配置需传入容器，仅在宿主 shell 设置无效。

容器启动后，在宿主机执行以下只读检查（标准版将容器名替换成实际名称）：

```bash
docker exec vllm-sm75-ultra nvidia-smi
docker exec vllm-sm75-ultra python3 -c 'import ctypes; ctypes.CDLL("libnvidia-api.so.1"); ctypes.CDLL("libnvidia-ml.so.1"); print("NVAPI / NVML load OK")'
```

NVAPI 找不到时检查宿主路径和只读挂载；NVML 缺失或 `nvidia-smi` 失败时检查宿主驱动、GPU 容器运行时及 `utility` 能力。库加载成功后，还需实际确认空闲进入 P8、负载恢复 16（驱动自动）。上游文档要求在宿主运行管理器；本项目提供容器内包装和宿主库挂载，属于本项目的集成方式，不能据此认定任意容器环境均受上游支持。同一 GPU 只运行一个电源管理实例。

### 模式与参数

| 模式 | 行为 | 显存 | 空闲功耗（4×T10 实测） | 唤醒 |
|---|---|---|---:|---|
| `sleep`（可选） | vLLM 空闲 30 分钟 exit 休眠（`AUTO_SLEEP_*` 可调） | 释放 | 10–15 W/卡 | 重建约 2 分钟 |
| `pstate`（默认） | 容器内监管器：GPU 低负载且 30 分钟无 API 请求 → P8；有请求/负载 → 16（驱动自动） | 常驻 | ≈12 W/卡（原 ≈40 W/卡） | 不重载模型；当前版本 TTFT 另行实测 |

```bash
POWER_MODE=pstate bash docker/run.sh
# 宿主库不在常见路径时显式指定：
PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1 POWER_MODE=pstate bash docker/run.sh
```

- 监管器在容器内读取同容器 `/metrics`（API 请求）与 `nvidia-smi`（GPU 负载），不感知 docker/宿主状态。
- 本项目显式挂载 `libnvidia-api.so.1`；标准版脚本自动探测宿主路径，找不到则退出 2。ultra 的指定方法见下方首次启动示例。
- 可调参数（透传容器）：`PSTATE_IDLE_TIMEOUT`（默认 1800 秒）、`PSTATE_UTIL`（默认 5%）、`PSTATE_CONFIRM`（默认 60 秒）、`PSTATE_GPUS`、`PSTATE_LOW/PSTATE_HIGH`（默认 8/16，禁止 0/0）、`PSTATE_POLL`。
- 同一时间只保留一个 P-State 管理实例；宿主若已有 nvidia-pstated 会提示冲突。
- `pstate` 模式下 `AUTO_SLEEP_*` 会被忽略并提示。

## 自动休眠与缓存目录

启动脚本默认常驻 P-State。选择 `POWER_MODE=sleep` 时保留自动休眠配置，默认 30 分钟 exit；60 秒休眠测试用 `POWER_MODE=sleep AUTO_SLEEP_IDLE_TIMEOUT=1`。模式要求、CPU RAM/磁盘预算、统一持久化挂载和旧目录迁移见[使用说明](../docs/sleep-and-cache.md)。旧布局不会自动迁移；先保留旧缓存并更新挂载，再运行新脚本。镜像名称仍是 `vllm-sm75:v$(cat docker/VERSION)`，重建镜像后必须重建容器才能生效。


## ultra 首次启动

```bash
ULTRA_DATA_ROOT=/path/to/ultra MODEL_ROOT=/path/to/models \
  PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1 \
  EDITION=ultra bash docker/run.sh
# 读取首次生成的 Web token，不需要预先配置 VLLM_API_KEY
docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

上面 `show` 命令在 **Docker 宿主机终端**执行，输出的一整行就是 Web 登录 token。默认容器名为 `vllm-sm75-ultra`；设置过 `CONTAINER_NAME` 时换成实际名称。token 保存在容器 `/data/key`，对应宿主机 `<ULTRA_DATA_ROOT>/console/key`。Unraid 的容器 Console 内直接执行 `node /opt/sm75-workbench/console/auth-cli.mjs show`。该 token 用于 Web 登录，不是模型 API key；重启和升级保留数据目录即可沿用。

使用 `http://<host>:1615` 登录，在模型库登记只读 `/models/<目录>`，创建配置后启动模型。下载目录默认为持久化的 `/data/models`，编译缓存为 `/data/cache`。已有模型 API key、Web token 和已保存设置不会被启动脚本覆盖。`ULTRA_DATA_ROOT` 下 console/home/workspace 和缓存分别挂载；不得用空目录替换现有数据完成所谓升级。

需要 P-State 时通过 `PSTATE_NVAPI_LIB=/宿主机/libnvidia-api.so.1` 挂载驱动库；未挂载时在控制台选择 sleep 电源模式。`GPUS`、`CONTAINER_NAME`、`CONSOLE_PORT`、`PORT`、`IMAGE` 可覆盖默认值。默认只启动管理服务，不自动加载模型。升级已有部署请按 [ultra 数据迁移/回退](../ultra/README.md) 保留原挂载、身份和配置。

## 检查入口

```bash
python3 -m unittest discover -s tests -p 'test_build_contract.py'
python3 -m unittest discover -s tests -p 'test_release_scripts.py'
python3 -m unittest discover -s tests -p 'test_regression_comparison.py'
node --test ultra/source/console/test/*.test.mjs
python3 tools/check-release.py
```

命令构造测试使用模拟 Docker，不会操作真实服务。真实 GPU 性能验证另用 `tools/benchmark-regression.py`，契约见 [发布回归审计](../docs/validation/2026-09-18-release-consolidation.md)。不能用静态检查替代推理、休眠/唤醒和性能门禁。

同一空闲引擎按固定配置采样（密钥文件仅本地读取，不写入结果）：

```bash
python3 tools/benchmark-regression.py --key-file /private/api-key \
  --model YOUR_SERVED_MODEL --label v015 --output /results/v015.jsonl
# 三个版本分别测量，文件顺序以 v0.1.4 基线在前；缺失样本、哈希不一致或吞吐下降 >5% 返回失败。
python3 tools/compare-regression.py /results/v014.jsonl /results/v015.jsonl /results/ultra.jsonl \
  --output /results/comparison.json
```

比较工具不自动切换容器，也不验证硬件/时钟/参数是否对齐；这些条件必须单独留证。输入除输出长度外，还需保持相同模型、tokenizer、TP、KV、调度和缓存策略。
