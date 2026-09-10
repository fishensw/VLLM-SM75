#!/usr/bin/env bash
# 容器内"软编译": 从挂载的项目目录(WORK, 默认 /work) 把 overlay 应用到镜像。
# 流程: overlay .py(秒级) -> speculative(6 .py) -> flashqla.so(hash 门控,
#       base 已预编通常跳过)。
# 幂等: 只改 .py 时秒级完成; 只有 .cu 变了才触发对应 nvcc 重编。
# 用法(容器内):  bash /opt/vllm-sm75/fast_compile.sh   然后 docker restart <容器>
set -euo pipefail

WORK="${WORK:-/work}"
EXT=/opt/vllm-sm75/extensions
EV=/opt/vllm-sm75/evidence
VLLM_PKGS=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
QLA_DIR="$WORK/vllm/third_party/flash_qla_sm75"
mkdir -p "$EXT" "$EV" /opt/vllm-sm75/work
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS="${MAX_JOBS:-1}"

log() { echo "[fast_compile] $*"; }

# --- 1) overlay .py: 整文件覆盖到 site-packages(改 .py 秒级生效) ---
# --source-copy 生成 /opt/vllm-sm75/source/v0.1.4/vllm, 供 FAST_VERIFY=1 的 AST 校验
log "应用 overlay .py (-> $VLLM_PKGS)"
python3 /opt/vllm-sm75/install_sm75_overlay.py "$WORK/vllm" \
    --source-copy /opt/vllm-sm75/source/v0.1.4/vllm

# --- 2) speculative: 拷 6 个 .py 并装到 vllm 包(model_runner hook 校验) ---
if [ -d "$WORK/docker/speculative" ]; then
  log "应用 speculative overlay"
  rm -rf /opt/vllm-sm75/speculative
  cp -a "$WORK/docker/speculative" /opt/vllm-sm75/speculative
  python3 /opt/vllm-sm75/install_speculative.py
else
  log "无 $WORK/docker/speculative, 跳过 speculative(非 MTP/DFlash2 模式可)"
fi

# --- 3) flashqla.so: base 已预编; 只有 gdn_forward.cu 变了才重编 ---
qla_src_hash() { sha256sum "$QLA_DIR/csrc/gdn_forward.cu" | awk '{print $1}'; }
qla_now=$(qla_src_hash)
qla_prev=$(cat "$EV/flashqla.hash" 2>/dev/null || echo "")
if [ "$qla_now" != "$qla_prev" ] || [ ! -s "$EXT/flash_qla_sm75_gdn.so" ]; then
  log "编译 flash_qla_sm75_gdn.so (源码变化或无产物)"
  python3 "$QLA_DIR/build_extension.py" \
    --build-directory /opt/vllm-sm75/work/qla --output "$EXT/flash_qla_sm75_gdn.so" --verbose
  cuobjdump --list-elf "$EXT/flash_qla_sm75_gdn.so" > "$EV/flashqla_cuobjdump.txt"
  test "$(grep -Eo 'sm_[0-9]+' "$EV/flashqla_cuobjdump.txt" | sort -u)" = "sm_75"
  echo "$qla_now" > "$EV/flashqla.hash"
  log "flash_qla_sm75_gdn.so 编译完成"
else
  log "flashqla.so 未变化(base 预编), 跳过"
fi

# --- 4) 可选契约校验 ---
if [ "${FAST_VERIFY:-0}" = "1" ]; then
  log "运行 verify_sm75_image.py"
  python3 /opt/vllm-sm75/verify_sm75_image.py
fi

log "软编译完成。下一步: docker restart <容器> 再 serve。"
log "  .so 产物: $EXT/flash_qla_sm75_gdn.so"
log "  (firefly.cu 仍走运行时 JIT ~2s, 不在此编译)"
