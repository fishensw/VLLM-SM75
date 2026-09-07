# v0.1.3 独立构建与启动

v0.1.3 在 v0.1.2 基础上加入自动休眠与透明唤醒，见[更新说明](../docs/releases/v0.1.3.zh-CN.md)。

## 1. 准备

使用 Linux x86_64、Docker、Git、Bash；启动需要 NVIDIA 驱动及 NVIDIA Container Toolkit。四卡示例按 TP4 配置。构建下载官方镜像和公开依赖，仅编译 SM75 扩展，默认 `MAX_JOBS=1`。

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
```

下述文件需要从 v0.1.3 发布版本取得。

## 2. 构建统一镜像

```bash
bash docker/build.sh
```

基础环境直接使用官方 `vllm/vllm-openai:v0.28.0-cu129`，固定 amd64 digest `sha256:50509e700235cea487715cedeb501d20a1cd15fa6a54ce93688284bd0d96995d`。Dockerfile 安装 FlashInfer 0.6.18、移除不适用的 JIT cache 包，再安装本仓库适配、编译 FlashQLA 并运行构建检查。

产物：`vllm-sm75:v0.1.3`，包含普通推理、MTP、DFlash2 和自动休眠支持，由启动参数选择模式。

## 3. 启动

```bash
export VLLM_API_KEY='replace-with-your-api-key'
# 替换成宿主机实际绝对路径；保存模型下载、vLLM 编译和 FlashInfer 缓存。
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75-cache
VARIANT=base FORMAT=fp8 bash docker/run.sh
```

MTP5 使用 `VARIANT=mtp`，要求模型具有匹配 MTP 权重。普通和 MTP 使用自动 KV；FP8 使用 seq4/batch8192，AWQ 使用 seq8/batch16384，utilization 均为0.87、max-model-len=auto。

AWQ 或 DFlash 需要先下载模型及匹配 draft，放在自选目录，再只读挂载：

```bash
export MODEL_ROOT=/path/to/downloaded-models
mkdir -p "$MODEL_ROOT"
# 通过镜像自带的 CLI 下载，无需在宿主机安装 Python 环境。
docker run --rm --volume "$MODEL_ROOT:/models" --entrypoint modelscope \
vllm-sm75:v0.1.3 download --model incoai/Qwen3.8-27B-DFlash2 \
  --local_dir /models/Qwen3.8-27B-DFlash2
docker run --rm --volume "$MODEL_ROOT:/models" --entrypoint hf \
  vllm-sm75:v0.1.3 download philbert440/Qwen3.8-27B-W4A16-AWQ \
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
