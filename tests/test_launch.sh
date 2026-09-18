#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export IMAGE="vllm-sm75:v$(tr -d ' \r\n' < "$ROOT/docker/VERSION")"
export VLLM_API_KEY=configuration-test-placeholder
export VLLM_SM75_CACHE_ROOT="$(mktemp -d -t sm75-launch-test-XXXXXXXX)"
export MODEL_ROOT="$VLLM_SM75_CACHE_ROOT"
export MODEL=/models/test-model DRAFT_MODEL=/models/test-draft
docker() {
  local args=" $* "
  [[ "$args" != *" $IMAGE serve "* ]]
  if [[ "$FORMAT" == awq ]]; then
    [[ "$args" == *'--max-num-seqs 8 '* ]]
    [[ "$args" == *'--max-num-batched-tokens 16384 '* ]]
    [[ "$args" == *'--gpu-memory-utilization 0.87 '* ]]
    [[ "$args" == *'/awq/vllm:/root/.cache/vllm'* ]]
    [[ "$args" == *'/awq/triton:/root/.triton/cache'* ]]
    if [[ "$VARIANT" == dflash2 ]]; then
      [[ "$args" == *'--kv-cache-memory-bytes 4294967296 '* ]]
      [[ "$args" == *'"num_speculative_tokens":7'* ]]
      [[ "$args" == *'"cpu_bytes_to_use":8589934592'* ]]
    fi
  fi
  [[ "$args" == *" $IMAGE "* ]]
  [[ "$args" != *"$IMAGE-mtp"* && "$args" != *"$IMAGE-dflash2"* ]]
  if [[ "$VARIANT" == base ]]; then
    [[ "$args" != *'--speculative-config'* ]]
  else
    [[ "$args" == *'--speculative-config'* ]]
  fi
  if [[ "$VARIANT" == dflash2 ]]; then
    [[ "$args" == *'--kv-cache-memory-bytes'* ]]
  else
    [[ "$args" != *'--kv-cache-memory-bytes'* ]]
    [[ "$args" == *'--max-model-len auto'* ]]
  fi
  if [[ "${POWER_MODE:-pstate}" == pstate ]]; then
    [[ "$args" == *'--entrypoint /opt/vllm-sm75/pstate-entrypoint.sh '* ]]
    [[ "$args" == *'--env PSTATE_IDLE_TIMEOUT=1800 '* ]]
    [[ "$args" == *'--env PSTATE_UTIL=5 '* ]]
    [[ "$args" == *'--env PSTATE_CONFIRM=60 '* ]]
    [[ "$args" == *'--env PSTATE_LOW=8 '* ]]
    [[ "$args" == *'--env PSTATE_HIGH=16 '* ]]
    [[ "$args" == *'--env PSTATE_POLL=5 '* ]]
    [[ "$args" == *'--env PSTATE_GPUS=0,1,2,3 '* ]]
    [[ "$args" == *'--auto-sleep-idle-timeout 0 '* ]]
    [[ "$args" != *'--auto-sleep-offload-target'* ]]
    [[ "$args" == *"$PSTATE_NVAPI_LIB:/usr/local/nvidia/lib64/libnvidia-api.so.1:ro"* ]]
  fi
}
export -f docker
for VARIANT in base mtp dflash2; do
  for FORMAT in fp8 awq; do
    export VARIANT FORMAT
    POWER_MODE=sleep AUTO_SLEEP_IDLE_TIMEOUT=0 bash "$ROOT/docker/run.sh"
    printf 'PASS %s %s unified-image arguments\n' "$VARIANT" "$FORMAT"
  done
done
export VARIANT=base FORMAT=fp8
PSTATE_LIB="$(mktemp -t sm75-nvapi-XXXXXXXX.so.1)"
export PSTATE_NVAPI_LIB="$PSTATE_LIB"
unset POWER_MODE
bash "$ROOT/docker/run.sh"
printf 'PASS default pstate mode arguments\n'
rm -f "$PSTATE_LIB"
unset PSTATE_NVAPI_LIB
unset POWER_MODE
if POWER_MODE=bogus bash "$ROOT/docker/run.sh" >/dev/null 2>&1; then
  echo 'FAIL invalid POWER_MODE accepted'
  exit 1
else
  printf 'PASS invalid POWER_MODE rejected\n'
fi
