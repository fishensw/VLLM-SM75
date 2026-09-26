#!/usr/bin/env bash
# Internal implementation; public entry: EDITION=ultra bash docker/run.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d ' \r\n' < "$SCRIPT_DIR/VERSION")"
: "${ULTRA_DATA_ROOT:?Set ULTRA_DATA_ROOT to an absolute persistent host directory}"
: "${MODEL_ROOT:?Set MODEL_ROOT to your existing absolute model directory}"
[[ "$ULTRA_DATA_ROOT" == /* && "$ULTRA_DATA_ROOT" != / && "$MODEL_ROOT" == /* && -d "$MODEL_ROOT" ]] || {
  echo 'Use a dedicated absolute data directory and an existing absolute model directory' >&2; exit 2;
}
cache="${VLLM_SM75_CACHE_ROOT:-$ULTRA_DATA_ROOT/cache}"
[[ "$cache" == /* && "$cache" != / ]] || exit 2
mkdir -p "$ULTRA_DATA_ROOT/console" "$ULTRA_DATA_ROOT/home" "$ULTRA_DATA_ROOT/workspace" "$cache"
extra=()
if [[ -n "${PSTATE_NVAPI_LIB:-}" ]]; then
  [[ -f "$PSTATE_NVAPI_LIB" && "$PSTATE_NVAPI_LIB" == /* ]] || exit 2
  extra+=(--volume "$PSTATE_NVAPI_LIB:/usr/local/nvidia/lib64/libnvidia-api.so.1:ro")
fi
# Existing data is neither migrated nor recursively chowned by this launcher.
# The manager creates its own files; DSH drops to UID/GID 1000 itself.
exec docker run --detach --name "${CONTAINER_NAME:-vllm-sm75-ultra}" \
  --gpus "${GPUS:-all}" --shm-size 16g --ulimit nofile=1048576:1048576 \
  --cap-drop ALL --cap-add CHOWN --cap-add FOWNER --cap-add DAC_OVERRIDE \
  --cap-add SETUID --cap-add SETGID --cap-add KILL --security-opt no-new-privileges \
  --publish "${CONSOLE_PORT:-1615}:1615" --publish "${PORT:-8000}:8000" \
  --env SM75_CONSOLE_ROOT=/data --env SM75_SINGLE_CONTAINER=1 \
  --env NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL-SYS}" \
  --env MAX_JOBS=1 --env NVCC_THREADS=1 \
  --volume "$ULTRA_DATA_ROOT/console:/data" --volume "$cache:/data/cache" \
  --volume "$ULTRA_DATA_ROOT/home:/dsh/home" --volume "$ULTRA_DATA_ROOT/workspace:/dsh/workspace" \
  --volume "$MODEL_ROOT:/models:ro" "${extra[@]}" \
  "${IMAGE:-vllm-sm75:v${VERSION}-ultra}"
