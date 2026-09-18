#!/usr/bin/env bash
# 容器内 P-State 启动包装（POWER_MODE=pstate）：
#   - 后台运行 pstate-supervisor.sh（30 分钟无请求 + 低负载 -> P8；有请求 -> 16 自动）；
#   - 前台运行 vllm serve；容器停止时先让监督器恢复 16 再退出。
set -euo pipefail

PSTATE_PID=""
if command -v pstate-supervisor.sh >/dev/null 2>&1; then
  pstate-supervisor.sh &
  PSTATE_PID=$!
  echo "[pstate] supervisor started pid=$PSTATE_PID"
elif [[ -x /opt/vllm-sm75/pstate-supervisor.sh ]]; then
  /opt/vllm-sm75/pstate-supervisor.sh &
  PSTATE_PID=$!
  echo "[pstate] supervisor started pid=$PSTATE_PID"
else
  echo "[pstate] pstate-supervisor.sh not found in image" >&2
fi

vllm serve "$@" &
VLLM_PID=$!

on_term() {
  if [[ -n "$PSTATE_PID" ]] && kill -0 "$PSTATE_PID" 2>/dev/null; then
    kill -INT "$PSTATE_PID" 2>/dev/null || true
  fi
  kill -TERM "$VLLM_PID" 2>/dev/null || true
}
trap on_term TERM INT

wait "$VLLM_PID"; rc=$?
if [[ -n "$PSTATE_PID" ]] && kill -0 "$PSTATE_PID" 2>/dev/null; then
  kill -INT "$PSTATE_PID" 2>/dev/null || true
fi
exit "$rc"
