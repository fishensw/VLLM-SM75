#!/usr/bin/env bash
# 容器内 P-State 监督器（POWER_MODE=pstate）：
#   有 GPU 负载或近期有 API 请求 -> nvidia-pstated 强制 16（交还驱动自动控制）；
#   低负载且 PSTATE_IDLE_TIMEOUT 秒（默认 1800）无请求 -> 切到 P8，有负载自动升回。
# 只依赖容器内的 nvidia-pstated / nvidia-smi / curl（读同容器 /metrics），不感知 docker。
set -uo pipefail

GPU_IDS="${PSTATE_GPUS:-0,1,2,3}"
PSTATE_HIGH="${PSTATE_HIGH:-16}"
PSTATE_LOW="${PSTATE_LOW:-8}"
IDLE_TIMEOUT="${PSTATE_IDLE_TIMEOUT:-1800}"
UTIL_THRESHOLD="${PSTATE_UTIL:-5}"
CONFIRM="${PSTATE_CONFIRM:-60}"
POLL="${PSTATE_POLL:-5}"
METRICS_URL="${PSTATE_METRICS_URL:-http://127.0.0.1:8000/metrics}"

if [[ "$PSTATE_LOW" == 0 && "$PSTATE_HIGH" == 0 ]]; then
  echo "[pstate] refusing -psl 0 -psh 0 (pins T10 at ~645MHz)" >&2
  exit 2
fi
if ! command -v nvidia-pstated >/dev/null 2>&1; then
  echo "[pstate] nvidia-pstated not found in image" >&2
  exit 1
fi

echo $$ > /tmp/pstate-supervisor.pid
mode=""
pstate_pid=""
last_activity="$(date +%s)"
idle_since=0
ready=0
warned_metrics=0
warned_util=0

stop_pstated() {
  if [[ -n "$pstate_pid" ]] && kill -0 "$pstate_pid" 2>/dev/null; then
    kill -INT "$pstate_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pstate_pid" 2>/dev/null || break; sleep 0.1; done
  fi
  pstate_pid=""
}

start_pstated() { # $1: active|idle
  local low high
  if [[ "$1" == active ]]; then low="$PSTATE_HIGH"; high="$PSTATE_HIGH"; else low="$PSTATE_LOW"; high="$PSTATE_HIGH"; fi
  nvidia-pstated -i "$GPU_IDS" -psl "$low" -psh "$high" -ut 0 -ibs 1 -si 100 &
  pstate_pid=$!
  mode="$1"
  echo "[pstate] mode=$1 (-psl $low -psh $high)"
}

set_mode() {
  [[ "$mode" == "$1" ]] && return 0
  stop_pstated
  start_pstated "$1"
}

max_util() {
  local u
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_IDS" 2>/dev/null \
      | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1) || return 1
  [[ -n "$u" ]] || return 1
  echo "$u"
}

metrics_total() {
  local t
  t=$(curl -s -m 3 "$METRICS_URL" 2>/dev/null | awk '
      /^vllm:(prompt_tokens_total|generation_tokens_total|request_success_total)/ { s += $NF }
      END { if (NR > 0) printf "%d", s + 0 }') || return 1
  [[ -n "$t" ]] || return 1
  echo "$t"
}

stop_requested=0
trap 'stop_requested=1; stop_pstated; rm -f /tmp/pstate-supervisor.pid /tmp/pstate-state.json' TERM INT

STATE_FILE=/tmp/pstate-state.json
echo "[pstate] supervisor start: idle_timeout=${IDLE_TIMEOUT}s util<${UTIL_THRESHOLD}% confirm=${CONFIRM}s poll=${POLL}s"
start_pstated active
last_total=""
while [[ "$stop_requested" == 0 ]]; do
  sleep "$POLL"
  now=$(date +%s)

  if [[ -f /tmp/pstate-command.json ]]; then
    cmd=$(sed -n 's/.*"action"[[:space:]]*:[[:space:]]*"\([a-z0-9]*\)".*/\1/p' /tmp/pstate-command.json 2>/dev/null)
    rm -f /tmp/pstate-command.json
    if [[ "$cmd" == "p8" ]]; then
      set_mode idle
      idle_since=$now
      last_activity=$(( now - IDLE_TIMEOUT ))
      echo "[pstate] manual P8 requested; holding until activity"
    fi
  fi

  total=$(metrics_total) || total=""
  if [[ -n "$total" ]]; then
    if (( ready == 0 )); then
      ready=1
      last_activity=$now
      echo "[pstate] engine ready; idle countdown starts now"
    fi
    if [[ -n "$last_total" && "$total" != "$last_total" ]]; then
      last_activity=$now
    fi
    last_total="$total"
  else
    if (( ready == 1 )); then
      ready=0
      last_total=""
      echo "[pstate] metrics unavailable; holding driver auto until engine is ready again"
    fi
    if (( warned_metrics == 0 )); then
      warned_metrics=1
      echo "[pstate] metrics not ready yet; idle timer will start after first successful read"
    fi
  fi

  util=$(max_util) || util=""
  if [[ -z "$util" ]]; then
    if (( warned_util == 0 )); then
      warned_util=1
      echo "[pstate] gpu utilization unavailable; treating as busy" >&2
    fi
    util=100
  fi
  if (( util >= UTIL_THRESHOLD )); then
    last_activity=$now
  fi

  next=$(( IDLE_TIMEOUT - (now - last_activity) )); (( next < 0 )) && next=0
  printf '{"mode":"%s","idle_timeout":%s,"idle_remaining":%s,"util":%s,"updated":%s}\n' "$mode" "$IDLE_TIMEOUT" "$next" "$util" "$now" > "$STATE_FILE" 2>/dev/null || true
  if (( ready == 1 )) && (( now - last_activity >= IDLE_TIMEOUT )); then
    if (( idle_since == 0 )); then
      idle_since=$now
      echo "[pstate] idle condition reached (util=${util}%, no request for $((now - last_activity))s); confirming ${CONFIRM}s"
    fi
    if (( now - idle_since >= CONFIRM )); then
      set_mode idle
    fi
  else
    if (( idle_since != 0 )); then echo "[pstate] activity detected; back to driver auto (16)"; fi
    idle_since=0
    set_mode active
  fi
done
stop_pstated
