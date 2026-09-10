#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$(mktemp -d -t sm75-sleep-launch-XXXXXXXX)"
trap 'rm -rf -- "$test_root"' EXIT
export VLLM_API_KEY=configuration-test-placeholder
export VLLM_SM75_CACHE_ROOT="$test_root/cache with space"
export VLLM_SM75_MODEL_CACHE_ROOT="$test_root/model files"
export VARIANT=base FORMAT=fp8
export AUTO_SLEEP_RELOAD_PATH=/models/checkpoint
export AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL=0
has_arg() { local needle="$1"; shift; for arg in "$@"; do [[ "$arg" != "$needle" ]] || return 0; done; return 1; }
docker() {
  has_arg "$VLLM_SM75_CACHE_ROOT/$FORMAT/vllm:/root/.cache/vllm" "$@"
  has_arg "$VLLM_SM75_CACHE_ROOT/shared/flashinfer:/root/.cache/flashinfer" "$@"
  has_arg "$VLLM_SM75_MODEL_CACHE_ROOT:/root/.cache/modelscope" "$@"
  has_arg "$VLLM_SM75_MODEL_CACHE_ROOT:/root/.cache/huggingface" "$@"
  has_arg vllm-sm75:v0.1.4 "$@"
  if [[ "$AUTO_SLEEP_IDLE_TIMEOUT" == 0 ]]; then
    ! has_arg --auto-sleep-idle-timeout "$@"
    ! has_arg --enable-sleep-mode "$@"
  else
    has_arg --auto-sleep-idle-timeout "$@"
    has_arg --auto-sleep-reload-path "$@"
    has_arg /models/checkpoint "$@"
    has_arg --auto-sleep-page-cache-keep-interval "$@"
    if [[ "$AUTO_SLEEP_OFFLOAD_TARGET" == exit ]]; then
      ! has_arg --enable-sleep-mode "$@"
    else
      has_arg --enable-sleep-mode "$@"
    fi
  fi
}
export -f docker has_arg
for AUTO_SLEEP_OFFLOAD_TARGET in exit cpu reload; do
  export AUTO_SLEEP_OFFLOAD_TARGET AUTO_SLEEP_IDLE_TIMEOUT=1
  bash "$ROOT/docker/run.sh"
  printf 'PASS %s required flags and persistent mounts\n' "$AUTO_SLEEP_OFFLOAD_TARGET"
done
export AUTO_SLEEP_IDLE_TIMEOUT=0
bash "$ROOT/docker/run.sh"
printf 'PASS disabled sleep\n'
export AUTO_SLEEP_IDLE_TIMEOUT=1 AUTO_SLEEP_OFFLOAD_TARGET=invalid
if bash "$ROOT/docker/run.sh" >/dev/null 2>&1; then echo 'invalid mode accepted' >&2; exit 1; fi
printf 'PASS invalid mode rejected\n'
