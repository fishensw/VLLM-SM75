#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_ROOT:?Set MODEL_ROOT to the parent of fp8ple}"
: "${CACHE_ROOT:?Set CACHE_ROOT to a dedicated cache directory}"
: "${VLLM_API_KEY:?Set VLLM_API_KEY for API authentication}"
mkdir -p "$CACHE_ROOT"
docker run -d --name "${CONTAINER_NAME:-vllm-sm75-next-ultra-0924}" \
  --gpus all --restart no --shm-size 64g --memory 240g --memory-swap 240g \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -p "${API_BIND:-0.0.0.0}:${API_PORT:-18001}:8000" \
  --mount "type=bind,src=$MODEL_ROOT,dst=/models,readonly" \
  --mount "type=bind,src=$CACHE_ROOT,dst=/cache" \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton \
  --entrypoint python3 "${IMAGE:-vllm-sm75-next-ultra-0924:latest}" /opt/flashnext/serve.py \
  --api-key "$VLLM_API_KEY" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"model":"/models/fp8ple/runtime/mtp-int4-g32"}'
