# v0.1.4 独立构建与启动

v0.1.3 在 v0.1.2 基础上加入自动休眠与透明唤醒，见[更新说明](../docs/releases/v0.1.3.zh-CN.md)。

## 1. 准备

使用 Linux x86_64、Docker、Git、Bash；启动需要 NVIDIA 驱动及 NVIDIA Container Toolkit。四卡示例按 TP4 配置。构建下载官方镜像和公开依赖，仅编译 SM75 扩展，默认 `MAX_JOBS=1`。

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
```

下述文件需要从 v0.1.4 发布版本取得。

## 2. 构建统一镜像

```bash
bash docker/build.sh
```

基础环境直接使用官方 `vllm/vllm-openai:v0.29.0-cu129`，固定 amd64 digest `sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2`。Dockerfile 安装 FlashInfer 0.6.18、移除不适用的 JIT cache 包，再安装本仓库适配、编译 FlashQLA 并运行构建检查。

产物：`vllm-sm75:v0.1.4`，包含普通推理、MTP、DFlash2 和自动休眠支持，由启动参数选择模式。

### 快速构建（本地使用，`Dockerfile.fast`）

统一镜像把 overlay 安装、FlashQLA 编译、契约校验都打进镜像，自包含、可共享/发布；但改代码反复构建时，`uv pip install transformers` 这类海量小文件的层在部分 containerd/overlayfs 环境偶发导出失败（`lease does not exist` / `lstat ...: no such file`）。

快速构建产出**本地专用、不对外共享**的镜像：它**不含** overlay `.py`，靠挂载本仓库目录 + 容器内软编译，因此只适合项目目录就在本机的本地使用（含开发迭代），不能像统一镜像那样拉走分发。

```bash
bash docker/build_fast.sh    # 产物 vllm-sm75-fast:v0.1.4，只含底座+FlashInfer+transformers+patch+预编译 FlashQLA
```

启动后挂载本仓库目录，容器内软编译（只改 `.py` 秒级；只有 `.cu` 变了才触发对应 nvcc 重编）：

```bash
docker run -d --name dev --gpus all -p 8000:8000 -v <本仓库>:/work vllm-sm75-fast:v0.1.4 sleep infinity
docker exec dev bash /opt/vllm-sm75/fast_compile.sh   # 应用 overlay + 软编译 FlashQLA
docker restart dev                                   # overlay/.so 进容器文件系统, 再 serve
docker exec -d dev vllm serve /data/<model> ...      # 或改 entrypoint 直接 serve
```

flash_qla 在 fast base 已预编译；`firefly.cu` 仍走运行时 JIT（~2s）。改 `.cu` 后重跑 `fast_compile.sh` + `restart` 即可。发版仍以 `build.sh` 统一镜像为准。

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
vllm-sm75:v0.1.4 download --model incoai/Qwen3.8-27B-DFlash2 \
  --local_dir /models/Qwen3.8-27B-DFlash2
docker run --rm --volume "$MODEL_ROOT:/models" --entrypoint hf \
  vllm-sm75:v0.1.4 download philbert440/Qwen3.8-27B-W4A16-AWQ \
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

## 自动休眠与缓存目录

启动脚本默认 30 分钟 exit，60 秒测试用 `AUTO_SLEEP_IDLE_TIMEOUT=1`。模式要求、CPU RAM/磁盘预算、统一持久化挂载和旧目录迁移见[使用说明](../docs/sleep-and-cache.md)。旧布局不会自动迁移；先保留旧缓存并更新挂载，再运行新脚本。镜像名称仍是 `vllm-sm75:v0.1.4`，重建镜像后必须重建容器才能生效。
