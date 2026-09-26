#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_ROOT:?Set MODEL_ROOT to the parent of fp8ple}"
: "${CACHE_ROOT:?Set CACHE_ROOT to a dedicated cache directory}"
: "${ULTRA_DATA_ROOT:?Set ULTRA_DATA_ROOT to a dedicated Next panel data directory}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$CACHE_ROOT" "$ULTRA_DATA_ROOT/console" "$ULTRA_DATA_ROOT/home" "$ULTRA_DATA_ROOT/workspace"
docker run -d --name "${CONTAINER_NAME:-vllm-sm75-next-ultra-0924}" \
  -e NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL-SYS}" \
  --gpus all --restart no --shm-size 64g --memory "${MEMORY_LIMIT:-112g}" --memory-swap "${MEMORY_LIMIT:-112g}" \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -p "${API_BIND:-0.0.0.0}:${CONSOLE_PORT:-1615}:1615" \
  -p "${API_BIND:-0.0.0.0}:${API_PORT:-18001}:8000" \
  --mount "type=bind,src=$MODEL_ROOT,dst=/models,readonly" \
  --mount "type=bind,src=$CACHE_ROOT,dst=/data/cache" \
  --mount "type=bind,src=$ULTRA_DATA_ROOT/console,dst=/data" \
  --mount "type=bind,src=$ULTRA_DATA_ROOT/home,dst=/dsh/home" \
  --mount "type=bind,src=$ULTRA_DATA_ROOT/workspace,dst=/dsh/workspace" \
  --mount "type=bind,src=$REPO_ROOT/next-ultra.mjs,dst=/opt/flashnext/next-ultra.mjs,readonly" \
  -e SM75_CONSOLE_ROOT=/data -e SM75_SINGLE_CONTAINER=1 \
  --entrypoint /usr/local/bin/node "${IMAGE:-vllm-sm75-next-ultra-0924:latest}" /opt/flashnext/next-ultra.mjs
