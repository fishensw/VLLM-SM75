> v0.1.4 使用说明。本文历史实测按原日期保留；不等于 Firefly 整合版本已完成验收。新增对比见[验证记录](validation/v0.1.4.md)。

# 已验证推荐：FP8 DFlash2 + 30 分钟 exit

适用于本地验证的 **4 × Tesla T10 16 GiB、TP4、约 31 GiB 主机内存**，优先目标为空闲省电，并复用主模型和草稿模型编译缓存。其他 GPU/内存容量不能直接套用显存预算。镜像名称保持 `vllm-sm75:v0.1.4`。

## 参数与要求

| 项目 | 已验证值 |
| --- | --- |
| 主模型 | `Qwen/Qwen3.8-27B-FP8` |
| 草稿 | `incoai/Qwen3.8-27B-DFlash2`，完整文件持续挂载 |
| TP / 最大上下文 | `4` / `262144`；本轮缓存修复只做短请求验证，不代表重新验收 262K 长上下文 |
| 并发 / batch | `max-num-seqs=4` / `max-num-batched-tokens=8192` |
| GPU utilization / KV bytes | `0.92` / `3288334336`，显式 KV bytes 是实际 KV 预算控制项 |
| KV dtype / 输入 block size | `fp8_e4m3` / `32`；不要将运行时派生的 832/1664 写回启动参数 |
| DFlash draft tokens / TP / KV | `7` / `4` / `auto` |
| CUDA Graph | `FULL_AND_PIECEWISE`，capture sizes `[8]` |
| CPU KV offload | `8589934592` bytes（8 GiB），不是整模型 RAM 休眠备份 |
| 休眠 | `exit`，日常 `30` 分钟；实测睡眠循环先用 `1` 分钟 |
| 共享内存 / CPU | Docker shm 上限 16 GiB（不是预占 16 GiB RAM）；主机实测 governor 为用户设置的 ondemand，不由启动脚本修改 |
| 缓存 | 主模型/草稿/候选选择器 → `cache/fp8/vllm`；FlashInfer → `cache/shared/flashinfer` |

## 完整启动示例

以下保留正式配置的推理参数和功能环境变量。网络改为便于复现的端口映射；本地 Unraid 正式容器使用既有 br0 静态地址，两者不同时启动。已使用 Unraid 模板的用户只需核对这些值，不必另建同 GPU 容器。

```bash
# 必须替换模型/缓存目录，设置自己的 API key；草稿已完整下载到示例容器路径。
export VLLM_API_KEY='replace-with-your-api-key'
export MODEL_CACHE_ROOT=/path/to/existing-model-cache
export CACHE_ROOT=/path/to/vllm-sm75/cache
mkdir -p "$CACHE_ROOT/fp8/vllm" "$CACHE_ROOT/shared/flashinfer"

docker run --detach --name vllm-sm75-fp8-dflash2 \
  --gpus all --shm-size 16g --ulimit nofile=1048576:1048576 \
  --publish 8000:8000 \
  --volume "$MODEL_CACHE_ROOT:/root/.cache/modelscope" \
  --volume "$MODEL_CACHE_ROOT:/root/.cache/huggingface" \
  --volume "$CACHE_ROOT/fp8/vllm:/root/.cache/vllm" \
  --volume "$CACHE_ROOT/shared/flashinfer:/root/.cache/flashinfer" \
  --env VLLM_USE_MODELSCOPE=true --env MODELSCOPE_CACHE=/root/.cache/modelscope/hub \
  --env VLLM_GDN_DECODE_KERNEL=triton --env VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --env VLLM_USE_NCCL_SYMM_MEM=0 --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env VLLM_ENABLE_CUDA_COMPATIBILITY=0 --env OMP_NUM_THREADS=2 \
  --env VLLM_ENGINE_READY_TIMEOUT_S=1800 --env VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
  --env VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  vllm-sm75:v0.1.4 Qwen/Qwen3.8-27B-FP8 \
  --served-model-name VLLM-Qwen3.8-27B --host 0.0.0.0 --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.92 \
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 3288334336 --block-size 32 \
  --dtype float16 --hf-overrides '{"dtype":"float16"}' --generation-config vllm \
  --attention-config '{"backend":"FLASHINFER"}' --gdn-prefill-backend flashqla_sm75 \
  --async-scheduling --enable-prefix-caching --enable-prompt-tokens-details \
  --disable-custom-all-reduce --enable-auto-tool-choice \
  --mamba-cache-mode align --mm-encoder-attn-backend TORCH_SDPA \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":8589934592}}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8]}' \
  --speculative-config '{"method":"dflash","model":"/root/.cache/huggingface/hub/models/incoai/Qwen3___8-27B-DFlash2","num_speculative_tokens":7,"draft_tensor_parallel_size":4,"max_model_len":262144,"kv_cache_dtype":"auto","attention_backend":"FLASHINFER","draft_sample_method":"probabilistic"}' \
  --auto-sleep-idle-timeout 30 --auto-sleep-offload-target exit
```

草稿路径必须对应真实下载位置；若放在其他挂载下，仅替换对应挂载及 JSON 中的 `model` 容器路径。首次使用可把末尾超时改成 `1`，验证后恢复 `30`。不要添加 `--enable-sleep-mode` 或改成 CPU 整模型备份来实现本推荐的 exit 省电效果。

## 已测效果与边界

日期：2026-09-08。修复镜像 ID `sha256:d70591b3d5e89c28a63ca3b7e1ffa08b5dfc466ffce1fd6b81381dce78e8827d`，部署时标签为 `vllm-sm75:v0.1.3`。他人同源码重建的镜像 ID 不要求相同。

| 项目 | 实测结果 |
| --- | --- |
| 60 秒 exit 休眠 | 请求完成后空闲计时；API health 200；四卡连续三次采样均 P8 |
| GPU 显存 | 驻留 13663 MiB/卡 → 休眠 3 MiB/卡 |
| 驻留待机功耗 | 单次快照 38.04–43.01 W/卡，四卡合计约 157.9 W |
| 休眠稳定功耗样本 | 9.97–15.31 W/卡；三次采样覆盖约 20 秒，不是长时间功耗平均 |
| 自动唤醒 | 两个并发短请求均成功，完整请求耗时 135.633 / 135.656 秒 |
| 编译缓存 | 已有缓存启动、exit 唤醒、改 30 分钟后启动，均 12 次 AOT 命中、0 次重新编译；三组键一致 |
| 唤醒缓存加载阶段 | 主模型 3.81 秒、草稿 0.76 秒、候选选择器 0.06 秒 |
| 正式 30 分钟配置启动 | 约 211.2 秒至 health 通过；健康和推理均 200，短算术请求 0.254 秒 |

30 分钟配置已正式运行，但本轮没有额外等待完整 30 分钟再重复整套睡眠验收。省电循环在相同推理参数、超时缩短到 60 秒的测试中验证。上述短请求不作为 decode 性能基准。

其他四个正式配置已保持原参数切换到同一修复镜像，但没有在本轮完成相同 GPU 验收，因此不列为“本次已验证推荐”。CPU/reload 模式也不冒充已经通过同等验证。模式资源要求和缓存迁移见[使用说明](sleep-and-cache.md)。
