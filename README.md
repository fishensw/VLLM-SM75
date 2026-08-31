# VLLM-SM75

项目目标是持续跟进官方
[vLLM](https://github.com/vllm-project/vllm)，完善其对 SM75 的兼容支持，
并优化相关内核。

当前 `vllm-sm75-v0.1.0` 基于官方 vLLM `v0.28.0`，已在 4 x Tesla T10
16 GiB、CUDA 12.9、TP4 环境中使用 Qwen3.8 27B FP8 完成验证。

## 功能与改进

- 新增 `flashqla_sm75` GDN prefill backend。
- 集成 image-owned FlashQLA-SM75 CUDA 扩展，只编译 `sm_75`，默认禁止
  runtime JIT，不依赖宿主机 `.so` 挂载。
- FlashInfer 从官方基线的 0.6.16.post3 升级到 0.6.18，并恢复 SM75
  full-attention 能力选择。
- GDN prefill 与 decode 独立控制；最终配置使用 FlashQLA-SM75 prefill 和
  Triton decode。
- 不支持的 GPU、dtype 或 GDN head dimension 自动回退到 Triton/FLA。
- 保留 Marlin FP8、FP8 E4M3 KV cache、prefix cache、async scheduling、
  `FULL_AND_PIECEWISE` CUDA Graph 和 CPU KV offload。
- FlashInfer sampler 在 SM75 上关闭，attention 仍使用 FlashInfer，sampling
  回退到 vLLM 原生实现。

FlashQLA 源码来自
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)，固定提交
`187b932dbd11940f0bcf52fb3675dd47fd69f313`。来源和许可证保留在
`vllm/third_party/flash_qla_sm75/`。

## 验证环境

| 组件 | 版本或配置 |
| --- | --- |
| VLLM-SM75 | `vllm-sm75-v0.1.0` |
| 上游 vLLM | `v0.28.0` / `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| GPU | 4 x Tesla T10 16 GiB / SM75 |
| CUDA | 12.9 |
| PyTorch | 2.13.0 |
| FlashInfer | 0.6.18，无 `flashinfer-jit-cache` |
| 验证模型 | Qwen3.8 27B FP8 |
| Tensor parallel | TP4 / PYNCCL |
| GDN | FlashQLA-SM75 prefill + Triton decode |

## 实测结果

测试保持模型、TP、attention、KV、sampling 和服务资源一致，仅比较 GDN
prefill 路线。

| 指标 | vLLM v0.28.0 生产基线 | `vllm-sm75-v0.1.0` | 变化 |
| --- | ---: | ---: | ---: |
| Cold TTFT | 1.9562 s | 1.7082 s | **提升 12.68%** |
| Prefix-cached TTFT | 0.5064 s | 0.4378 s | **提升 13.55%** |
| Decode | 39.6289 token/s | 39.6306 token/s | 基本持平 |
| Cached tokens | 1,568 | 1,568 | 不变 |

确定性输出、streaming、tool calling、reasoning parser、多模态、prefix cache、
FP8 KV 和 8 GiB CPU KV offload 均通过验证。候选 GPU KV 容量为 506,209
tokens，并观测到 821,297,152 bytes GPU 到 CPU KV 迁移。

## 构建镜像

准备兼容 vLLM 0.28.0、CUDA 12.9、PyTorch 2.13.0 和 SM75 的基础镜像：

```bash
export BASE_IMAGE='your-registry.example/vllm-openai:v0.28.0-cu129-sm75'

docker build \
  --file docker/Dockerfile.vllm-sm75-v0.1.0 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg BASE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --tag vllm-sm75-v0.1.0 \
  .
```

Dockerfile 会编译 SM75 扩展、检查 `cuobjdump` 架构、验证运行时版本，并在
构建阶段运行必要的 backend selector tests。

## 运行关键配置

在原有 vLLM serve 参数基础上使用：

```bash
export VLLM_GDN_DECODE_KERNEL=triton
export FLASH_QLA_SM75_ALLOW_JIT=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_NCCL_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

vllm serve /path/to/model \
  --tensor-parallel-size 4 \
  --attention-config '{"backend":"FLASHINFER"}' \
  --gdn-prefill-backend flashqla_sm75 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'
```

模型长度、并发、显存利用率、tool parser、reasoning parser 和多模态限制应按
实际模型及显存重新设置。

## License

vLLM 修改继续遵循上游 Apache-2.0 License。FlashQLA-SM75 文件保留其原始
MIT License 和来源说明。
