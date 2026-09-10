#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VLLM_API_KEY=configuration-test-placeholder
export VLLM_SM75_CACHE_ROOT="$(mktemp -d -t sm75-launch-test-XXXXXXXX)"
export MODEL_ROOT="$VLLM_SM75_CACHE_ROOT"
export MODEL=/models/test-model DRAFT_MODEL=/models/test-draft
docker() {
  local args=" $* "
  [[ "$args" == *' vllm-sm75:v0.1.4 serve '* ]]
  [[ "$args" != *'v0.1.4-mtp'* && "$args" != *'v0.1.4-dflash2'* ]]
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
}
export -f docker
for VARIANT in base mtp dflash2; do
  for FORMAT in fp8 awq; do
    export VARIANT FORMAT
    AUTO_SLEEP_IDLE_TIMEOUT=0 bash "$ROOT/docker/run.sh"
    printf 'PASS %s %s unified-image arguments\n' "$VARIANT" "$FORMAT"
  done
done
