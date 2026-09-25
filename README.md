# vllm-sm75-next-0924

基于 [vLLM-SM75 v0.1.6](https://github.com/fishensw/VLLM-SM75) Ultra 的 **Flash-Next 八卡专用分支**，面向 **8×Tesla T10 / NVIDIA Turing SM75**，提供 Qwen3.8-Flash-Next W4A16 FP8 PLE 的运行适配和推荐配置。

采用 **TP8＋Expert Parallel、FP16、FP8 KV、MTP5**，配置 **256K 总上下文与并发上限4**。保留 Ultra 控制台入口，提供可直接使用的 OpenAI 兼容推理 API 启动方式。

## 1. 适用范围

| 项目 | 配置 |
|---|---|
| GPU | 8×Tesla T10，每卡16GB，SM75 |
| 基础版本 | vllm-sm75 v0.1.6 Ultra / vLLM 0.30.0 |
| 主模型 | Qwen3.8-Flash-Next-W4A16-FP8PLE |
| 并行方式 | TP8＋EP |
| 推理精度 | 主模型FP16，W4专家权重，FP8 PLE与FP8 KV |
| MTP | 独立INT4 group32草稿，5步投机解码 |
| 上下文 | `max-model-len=262144`，包含输入与输出 |
| 并发 | `max-num-seqs=4`，面向单路256K及四路混合长度请求 |

本分支集中维护上述硬件与模型组合。四路并发不表示四个请求可以同时各占满256K。

## 2. Next 专用适配

本分支包含运行时补丁及配置模板，不能只将启动参数套用到未适配的原版镜像。

- **FP16兼容**：调整主模型和MTP中的HC参数精度，适配SM75运行路径。
- **FP8 PLE访问**：使用原始字节搬运处理PLE查询，保留原反量化逻辑。
- **B9放置策略**：PLE embedding表卸载到CPU，PLE key/value投影保留在GPU。
- **H1 HC优化**：对适用投影形状启用GEMV实现，其他形状保留回退路径。
- **MTP融合归一化**：减少MTP中两处归一化的临时显存分配。
- **MTP Graph配置**：采用PIECEWISE CUDA Graph，覆盖1/2/4路请求对应的验证批次。

当前沿用关闭QSA、跳过indexer的兼容路线，并包含继承的Torch编译适配；这些改动仅用于本分支隔离镜像，不作为所有SM75模型的通用默认。

## 3. 环境与模型准备

运行环境需要Linux x86_64、Docker、NVIDIA Container Toolkit，以及兼容Turing的NVIDIA驱动。八张GPU应由本服务独占使用。

主模型：[albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)。

按以下结构准备模型目录：

```text
/srv/models/
└── fp8ple/
    ├── config.json
    ├── tokenizer相关文件
    ├── 模型权重及索引文件
    └── runtime/
        └── mtp-int4-g32/
            ├── config.json
            ├── mtp-dense.safetensors
            └── mtp-routed-experts-int4.safetensors
```

目录示意只列出关键文件，部署时需保留完整模型及草稿文件。

**MTP草稿需要单独准备。** 主模型下载不包含本配置使用的全部草稿权重。当前配置对应本地RTN INT4 group32草稿，来源为 `RadixArk/Qwen3.8-Flash-Next-NVFP4` 的转换产物；本仓库不分发权重，也尚未提供完整的草稿转换流程。具体依赖见 [PROVENANCE.md](PROVENANCE.md)。

PLE表卸载和推理运行时需要主机内存。下方模板的 `240g` 是容器内存上限，不是固定预占或最低内存声明，应结合宿主容量配置。

## 4. 构建镜像

先按 [主项目构建说明](https://github.com/fishensw/VLLM-SM75#5-快速构建与启动)准备未应用本分支补丁的 `vllm-sm75:v0.1.6-ultra` 基础镜像，再在本分支目录执行：

```bash
docker build \
  --build-arg BASE_IMAGE=vllm-sm75:v0.1.6-ultra \
  -t vllm-sm75-next-ultra-0924:latest .
```

补丁安装会检查源码匹配情况。同名基础镜像标签可能变化，版本来源以 `PROVENANCE.md` 为准；不要对已经安装本分支补丁的镜像重复应用。

### 已完成的镜像验证

`vllm-sm75-next-ultra-0924:latest` 已完成构建，并从该镜像创建全新容器完成以下验证：

- 八张Tesla T10启动，TP8＋EP、MTP5配置生效。
- 单路256K总上下文：261632输入＋512输出。
- 并发4混合负载：1K／8K／32K／128K输入，各256输出。
- 上述容量验证无OOM、无请求抢占，服务健康检查通过。

这些验证在整理发布资料和README之前已经完成。仓库Dockerfile是随后对原分层构建流程的合并整理；该合并文件未单独重新执行，不影响上述最终镜像已完成验证的事实，也不代表逐字复现该文件已获验证。

## 5. 推荐启动：1615 面板＋Next 配置

首次启动自动创建面板配置 **Flash-Next TP8 · MTP5 · 256K · 并发4**，由面板自动启动和管理推理进程。使用独立的 Next 面板数据目录；已有其他 Ultra 数据目录不会被覆盖。

```bash
export MODEL_ROOT=/srv/models
export CACHE_ROOT=/srv/vllm-next/cache
export ULTRA_DATA_ROOT=/srv/vllm-next/ultra
export IMAGE=vllm-sm75-next-ultra-0924:latest
export API_BIND=0.0.0.0
export CONSOLE_PORT=1615
export API_PORT=18001
bash scripts/run-next.sh
```

脚本同时映射 **1615:1615** 和 **18001:8000**，挂载模型只读目录、独立缓存、面板数据及工作台目录。默认监听全部IPv4地址；请在防火墙中仅允许需要访问的网络。仅需本机访问时设置 `API_BIND=127.0.0.1`。

脚本将仓库内的 `next-ultra.mjs` 只读挂载到容器，并使用 Node 启动面板，可配合之前已构建的 Next 镜像使用。当前 Dockerfile 也已将该入口加入默认启动流程。不要再为面板模式指定 `--entrypoint python3`。

- **面板**：`http://<服务器IP>:1615`
- **推理 API Base URL**：`http://<服务器IP>:18001/v1`
- **模型名称**：`Flash-Next-TP8`
- **默认 MTP**：`{"method":"mtp","num_speculative_tokens":5,"model":"/models/fp8ple/runtime/mtp-int4-g32"}`

MTP草稿目录必须提前准备，见上方模型说明。面板可能先于推理服务就绪；首次编译及模型加载期间，请在面板查看引擎日志。

在 Docker 宿主机读取面板登录 token：

```bash
docker exec vllm-sm75-next-ultra-0924 node /opt/sm75-workbench/console/auth-cli.mjs show
```

推理 API Key 由面板独立生成和管理，登录后在 API 访问设置中查看或修改；它与面板登录 token 不同。面板模式无需设置 `VLLM_API_KEY`。

首次初始化从镜像内 `/opt/flashnext/selected-config.json` 导入完整推荐参数，仅将 `--model 路径` 转换为面板所需的位置参数形式。后续重启保留面板中的配置修改、默认配置、自动启动开关和凭据，不反复覆盖。默认不启用依赖额外驱动库的 P-State 管理，也未添加自动休眠参数。

服务检查：

```bash
curl -f http://127.0.0.1:1615/
curl -f http://127.0.0.1:18001/health
docker logs --tail 80 vllm-sm75-next-ultra-0924
```

仅需推理 API 时，可继续设置 `MODEL_ROOT`、`CACHE_ROOT`、`VLLM_API_KEY` 后运行 `bash scripts/run-next-api.sh`；该模式不启动1615面板。

## 6. 推荐参数

| 参数 | 推荐值 |
|---|---|
| `--dtype` / `--hf-overrides` | `float16` / `{"dtype":"float16"}` |
| `--tensor-parallel-size` | `8` |
| `--enable-expert-parallel` | 开启 |
| `--max-model-len` | `262144` |
| `--max-num-seqs` | `4` |
| `--max-num-batched-tokens` | `4096` |
| `--kv-cache-dtype` | `fp8_e4m3` |
| `--kv-cache-memory-bytes` | `2147483648`，每卡2GiB |
| `--gpu-memory-utilization` | `0.90` |
| `--block-size` | `32`，实际布局可能由引擎调整 |
| `--enable-prefix-caching` | 开启 |
| `--engram-config` | `{"cpu_offload":true}` |
| `--gdn-prefill-backend` | `flashqla_sm75` |

MTP配置：

```json
{
  "method": "mtp",
  "num_speculative_tokens": 5,
  "model": "/models/fp8ple/runtime/mtp-int4-g32"
}
```

Graph配置：

```json
{
  "mode": 0,
  "cudagraph_mode": "PIECEWISE",
  "cudagraph_capture_sizes": [1, 2, 4, 6, 12, 24]
}
```

镜像内置环境变量：

```text
OMP_NUM_THREADS=1
VLLM_SM75_QWEN38_HC_GEMV=1
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728
```

显式KV字节数控制KV分配，不能仅通过 `gpu-memory-utilization` 推断总显存占用。不要直接提高预填充批次或Graph捕获尺寸，运行时仍需预留显存。

## 7. 使用边界

- MTP收益取决于输入与草稿接受率，推荐配置不保证所有任务都更快；当前尚未实现prefill性能完全不退化。
- 长上下文和混合并发仍受KV及临时显存约束，长请求预填充可能影响短请求延迟。
- 当前路线关闭QSA并跳过indexer，不作原生QSA等价性承诺。
- 输出可能包含思考标签，调用方应按实际接口格式处理。
- 推荐参数和模型准备不能替代具体业务的质量与稳定性验收。

## 8. 项目文件

```text
Dockerfile                  镜像构建
patches/                    Flash-Next / SM75运行时适配
config/selected-config.json  八卡推荐配置
serve.py                    固化参数启动入口
scripts/run-next.sh          面板＋Next自动启动
scripts/run-next-api.sh      仅推理API启动
next-ultra.mjs               面板默认配置初始化入口
tests/next/                 Next GPU验证脚本
启动参数.md                 完整启动说明
PATCHES.md                  补丁范围与集成说明
PROVENANCE.md               版本和模型依赖
```

本分支README不包含性能榜单或原始测试数据。主项目的通用介绍、其它硬件与模型方案请参阅 [VLLM-SM75](https://github.com/fishensw/VLLM-SM75)。
