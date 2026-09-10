#!/usr/bin/env bash
set -euo pipefail
: "${VLLM_API_KEY:?Set VLLM_API_KEY to your own API key}"
VARIANT="${VARIANT:-base}"
FORMAT="${FORMAT:-fp8}"
case "$VARIANT" in base|mtp|dflash2) ;; *) echo 'VARIANT must be base, mtp, or dflash2' >&2; exit 2;; esac
case "$FORMAT" in fp8|awq) ;; *) echo 'FORMAT must be fp8 or awq' >&2; exit 2;; esac
if [[ "$FORMAT" == awq ]]; then
  : "${MODEL:?Set MODEL to your downloaded AWQ model directory under /models}"
else
  MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
fi
SERVE_NAME="${SERVED_MODEL_NAME:-VLLM-Qwen3.8-27B}"
CACHE_ROOT="${VLLM_SM75_CACHE_ROOT:?Set VLLM_SM75_CACHE_ROOT to an absolute host cache directory}"
[[ "$CACHE_ROOT" == /* ]] || { echo 'Cache path must be absolute' >&2; exit 2; }
MODEL_CACHE_ROOT="${VLLM_SM75_MODEL_CACHE_ROOT:-$CACHE_ROOT/models}"
[[ "$MODEL_CACHE_ROOT" == /* ]] || { echo 'Model cache path must be absolute' >&2; exit 2; }
mkdir -p "$MODEL_CACHE_ROOT" "$CACHE_ROOT/$FORMAT/vllm" "$CACHE_ROOT/shared/flashinfer" \
  "$CACHE_ROOT/$FORMAT/triton" "$CACHE_ROOT/shared/torch_extensions"
mounts=(--volume "$MODEL_CACHE_ROOT:/root/.cache/modelscope"
  --volume "$MODEL_CACHE_ROOT:/root/.cache/huggingface"
  --volume "$CACHE_ROOT/$FORMAT/vllm:/root/.cache/vllm"
  --volume "$CACHE_ROOT/shared/flashinfer:/root/.cache/flashinfer"
  --volume "$CACHE_ROOT/$FORMAT/triton:/root/.triton/cache"
  --volume "$CACHE_ROOT/shared/torch_extensions:/root/.cache/torch_extensions")
if [[ -n "${MODEL_ROOT:-}" ]]; then
  [[ "$MODEL_ROOT" == /* && -d "$MODEL_ROOT" ]] || { echo 'MODEL_ROOT must be an existing absolute directory' >&2; exit 2; }
  mounts+=(--volume "$MODEL_ROOT:/models:ro")
fi
seq=4; batch=8192; util=0.87; length=auto
[[ "$FORMAT" != awq ]] || { seq=8; batch=16384; }
image=vllm-sm75:v0.1.4
graph='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
extra=()
AUTO_SLEEP_IDLE_TIMEOUT="${AUTO_SLEEP_IDLE_TIMEOUT:-30}"
AUTO_SLEEP_OFFLOAD_TARGET="${AUTO_SLEEP_OFFLOAD_TARGET:-exit}"
if [[ "$AUTO_SLEEP_IDLE_TIMEOUT" != 0 ]]; then
  extra+=(--auto-sleep-idle-timeout "$AUTO_SLEEP_IDLE_TIMEOUT"
    --auto-sleep-offload-target "$AUTO_SLEEP_OFFLOAD_TARGET")
  case "$AUTO_SLEEP_OFFLOAD_TARGET" in
    cpu|reload|disk) extra+=(--enable-sleep-mode) ;;
    exit) ;;
    *) echo 'AUTO_SLEEP_OFFLOAD_TARGET must be cpu, reload, exit, or disk' >&2; exit 2 ;;
  esac
  if [[ "$AUTO_SLEEP_OFFLOAD_TARGET" == disk ]]; then
    mkdir -p "$CACHE_ROOT/sleep/$VARIANT-$FORMAT"
    mounts+=(--volume "$CACHE_ROOT/sleep/$VARIANT-$FORMAT:/sleep-state")
    extra+=(--auto-sleep-disk-path /sleep-state)
  fi
  if [[ -n "${AUTO_SLEEP_RELOAD_PATH:-}" ]]; then
    extra+=(--auto-sleep-reload-path "$AUTO_SLEEP_RELOAD_PATH")
  fi
  extra+=(--auto-sleep-page-cache-keep-interval "${AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL:-600}")
fi
if [[ "$VARIANT" == mtp ]]; then
  extra+=(--speculative-config '{"method":"mtp","num_speculative_tokens":5}')
  graph='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6]}'
elif [[ "$VARIANT" == dflash2 ]]; then
  : "${DRAFT_MODEL:?Set DRAFT_MODEL to the matching DFlash draft directory under /models}"
  [[ "$DRAFT_MODEL" == /models/* && -n "${MODEL_ROOT:-}" ]] || { echo 'Mount MODEL_ROOT and set DRAFT_MODEL=/models/your-draft' >&2; exit 2; }
  [[ "$DRAFT_MODEL" != *'"'* && "$DRAFT_MODEL" != *'\'* ]] || exit 2
  kv=3288334336; util=0.92; length=262144
  [[ "$FORMAT" != awq ]] || { kv=4294967296; util=0.87; length=auto; }
  extra+=(--kv-cache-memory-bytes "$kv" --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":4,\"max_model_len\":262144,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASHINFER\",\"draft_sample_method\":\"probabilistic\"}")
  graph='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8]}'
fi
docker run --detach --name "${CONTAINER_NAME:-vllm-sm75-$VARIANT-$FORMAT}" \
  --gpus all --shm-size 16g --ulimit nofile=1048576:1048576 \
  --publish "${PORT:-8000}:8000" "${mounts[@]}" \
  --env VLLM_USE_MODELSCOPE=true --env MODELSCOPE_CACHE=/root/.cache/modelscope/hub \
  --env VLLM_GDN_DECODE_KERNEL=triton --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env VLLM_USE_NCCL_SYMM_MEM=0 --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env TRITON_CACHE_DIR=/root/.triton/cache --env TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  --env VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}" \
  --env VLLM_FIREFLY="${VLLM_FIREFLY:-1}" --env VLLM_FIREFLY_AR="${VLLM_FIREFLY_AR:-auto}" \
  --env VLLM_FIREFLY_AR_BACKEND="${VLLM_FIREFLY_AR_BACKEND:-auto}" \
  --env VLLM_FIREFLY_AR_MIN_SIZE="${VLLM_FIREFLY_AR_MIN_SIZE:-1048576}" \
  --env OMP_NUM_THREADS=2 --env MAX_JOBS=1 --env TORCHINDUCTOR_COMPILE_THREADS=1 \
  "$image" "$MODEL" --served-model-name "$SERVE_NAME" \
  --host 0.0.0.0 --port 8000 --api-key "$VLLM_API_KEY" \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --max-num-seqs "$seq" --max-num-batched-tokens "$batch" \
  --gpu-memory-utilization "$util" --max-model-len "$length" \
  --attention-config '{"backend":"FLASHINFER"}' --gdn-prefill-backend flashqla_sm75 \
  --kv-cache-dtype fp8_e4m3 --block-size 32 --dtype float16 \
  --hf-overrides '{"dtype":"float16"}' --generation-config vllm \
  --enable-prefix-caching --async-scheduling --compilation-config "$graph" \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}' \
  "${extra[@]}"
