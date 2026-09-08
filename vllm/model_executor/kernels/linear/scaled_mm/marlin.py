# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 overlay: MarlinFP8ScaledMMLinearKernel + firefly(fp8→int8 prefill)混合。

全文件拷贝自上游 v0.28.0, 仅新增 firefly 分支(env 门控, off=上游行为):
  - process_weights_after_loading: repack 前快照干净 fp8 权重 [N,K] + block scale
    [N/128,K/128] + c_n; 不常驻 int8 副本(只有 hard, 单权重不爆显存)。
  - apply_weights: 大 M(prefill)走 int8 IMMA, 两种 hard 子模式:
      fused(VLLM_FIREFLY_FUSED=1): GEMM B-load 内即时反量化 fp8->int8;
      非 fused(VLLM_FIREFLY_FUSED=0): prefill 步 transient 反量化
      (fp8_dequant_only CUDA kernel; .so 不可用时回退 PyTorch fp8_to_int8) + int8 GEMM。
    小 M(decode) 或 firefly 禁用(fused .so 挂且 FUSED=1)走上游 weight-only fp8 Marlin。
  两种子模式数值逐 bit 一致(同反量化数学 + 同 int8 IMMA GEMM), 仅性能不同。

fp8 对称无 zp, 反量化比 AWQ 非对称 int4 简单。
见 vllm/model_executor/layers/quantization/utils/firefly.py。
"""

import logging
from collections.abc import Sequence

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.layers.quantization.utils.firefly import (
    _load_fused_mod,
    compute_c_n_fp8,
    fp8_fused_prefill_linear,
    fp8_to_int8,
    int8_prefill_linear,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    is_fp8_marlin_supported,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

logger = logging.getLogger(__name__)
_firefly_fused_logged = False
_firefly_unfused_logged = False
_firefly_fp8_fallback_logged = False


def _ff_fused_dbg_log(x: torch.Tensor) -> None:
    # 首次 fused prefill 触发时打一次点(确认走了 B-load dequant, 非静默回退 int8)。
    global _firefly_fused_logged
    if not _firefly_fused_logged:
        _firefly_fused_logged = True
        logger.info(
            "[firefly-fp8] FUSED GEMM prefill active (first M=%d)",
            x.numel() // x.shape[-1],
        )


def _ff_fp8_fallback_log(reason: str) -> None:
    # firefly prefill 回退上游 Marlin 时打一次明显日志(不刷屏):
    # fused .so 加载失败(无预编译且 JIT 挂)且 VLLM_FIREFLY_FUSED=1。
    global _firefly_fp8_fallback_logged
    if _firefly_fp8_fallback_logged:
        return
    _firefly_fp8_fallback_logged = True
    logger.warning(
        "[firefly-fp8] 回退上游 Marlin, firefly prefill 未启用 (%s)", reason
    )


def _ff_unfused_dbg_log(x: torch.Tensor, cuda_dequant: bool) -> None:
    # 首次非 fused hard prefill 触发时打一次点(确认走了 transient 反量化 + int8 GEMM,
    # 非静默回退)。标注反量化 kernel(CUDA fp8_dequant_only / PyTorch fp8_to_int8)。
    global _firefly_unfused_logged
    if not _firefly_unfused_logged:
        _firefly_unfused_logged = True
        logger.info(
            "[firefly-fp8] UNFUSED hard prefill active (first M=%d, %s dequant)",
            x.numel() // x.shape[-1],
            "CUDA" if cuda_dequant else "PyTorch",
        )


class MarlinFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """
    FP8 Marlin kernel for GPUs that lack FP8 hardware support.
    Leverages the Marlin kernel for fast weight-only FP8 quantization.

    SM75 overlay: 加 firefly int8 IMMA prefill(大 M, env 门控, off=上游行为)。
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        # Check if platform supports FP8 Marlin
        if not is_fp8_marlin_supported():
            return False, "FP8 Marlin requires compute capability 7.5 or higher"
        if envs.VLLM_BATCH_INVARIANT:
            return False, "FP8 Marlin not supported for batch invariant execution."
        if (
            compute_capability is not None
            and compute_capability >= 89
            and not envs.VLLM_TEST_FORCE_FP8_MARLIN
        ):
            return (
                False,
                "To apply FP8 Marlin on high-capability GPUs, please set "
                "VLLM_TEST_FORCE_FP8_MARLIN=1",
            )
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)
        self.marlin_input_dtype = None
        self.block_quant = self.config.weight_quant_key in {kFp8Static128BlockSym}
        self.size_k_first = not self.block_quant

    # ---- firefly 门控(fp8 权重 → int8 prefill) ----
    def _firefly_enabled(self) -> bool:
        # 仅对 128x128 block quant(有 per-block scale, weight_scale_inv)启用;
        # 非 block 的 fp8(tensor-wise/channel-wise, 无 weight_scale_inv)走上游,
        # 不反量化(fp8→int8 数学依赖 per-block scale 折叠)。
        return envs.VLLM_FIREFLY and self.block_quant

    def _firefly_m_large(self, x: torch.Tensor) -> bool:
        m = x.numel() // x.shape[-1]
        return m > envs.VLLM_FIREFLY_MIN_M

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # firefly-fp8: 只有 hard 模式——不常驻 int8 副本, prefill 步现反量化
        # fp8->int8(单权重, 不爆显存)。两种子模式:
        #   fused(VLLM_FIREFLY_FUSED=1): GEMM B-load 内即时反量化;
        #   非 fused(VLLM_FIREFLY_FUSED=0): prefill 步 transient 反量化 + int8 GEMM。
        # repack 前快照干净 fp8 权重 [N,K] + block scale [N/128,K/128](CUDA 上
        # process_fp8_weight_block_strategy 只 pad 权重、scale 原样返回, 快照即
        # 干净 canonical 布局) + c_n(两路共用)。
        # fused .so 挂 + FUSED=1 → 该层 firefly 禁用, 回退上游 Marlin。
        if self._firefly_enabled():
            w_fp8 = layer.weight.contiguous()  # [N, K] fp8 e4m3
            s_inv = layer.weight_scale_inv.contiguous()  # [N/128, K/128]
            c_n = compute_c_n_fp8(w_fp8, s_inv)  # [N] 两路共用
            layer._firefly_w_fp8 = w_fp8
            layer._firefly_s_inv = s_inv
            layer._firefly_c_n = c_n
            if envs.VLLM_FIREFLY_FUSED:
                if _load_fused_mod() is not None:
                    # fused: B-load 内即时反量化(不物化 w_int8, 省 1B/参数 int8 缓存)
                    layer._firefly_fused = True
                    layer._firefly_unfused = False
                else:
                    # fused .so 加载失败(无预编译 + JIT 挂) → 回退上游 Marlin
                    layer._firefly_fused = False
                    layer._firefly_unfused = False
                    _ff_fp8_fallback_log("fused .so 加载失败(无预编译且 JIT 回退也挂)")
            else:
                # 非 fused hard: prefill 步 transient 反量化 + int8 GEMM
                layer._firefly_fused = False
                layer._firefly_unfused = True
                # 反量化 kernel: 优先 CUDA(fp8_dequant_only, 在 fused .so 里),
                # .so 不可用时回退 PyTorch fp8_to_int8(正确但慢)。
                layer._firefly_unfused_cuda_dequant = _load_fused_mod() is not None

        if self.block_quant:
            weight, weight_scale_inv = process_fp8_weight_block_strategy(
                layer.weight, layer.weight_scale_inv
            )
            # Update layer with new values
            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale_inv", weight_scale_inv.data)
        # Non-block: callers must pass weight in (K, N) layout.

        layer.input_scale = None
        prepare_fp8_layer_for_marlin(
            layer, self.size_k_first, input_dtype=self.marlin_input_dtype
        )
        del layer.input_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # firefly-fp8: 大 M(prefill) 走 int8 IMMA, 两种 hard 子模式:
        #   fused: GEMM B-load 内反量化(fp8_fused_prefill_linear);
        #   非 fused: transient 反量化(fp8_dequant_only/fp8_to_int8) + int8 GEMM
        #   (int8_prefill_linear)。小 M(decode) 或 firefly 禁用(fused .so 挂且
        #   FUSED=1) 走上游 fp8 Marlin。两子模式数值逐 bit 一致。
        if self._firefly_enabled() and self._firefly_m_large(x):
            if getattr(layer, "_firefly_fused", False):
                _ff_fused_dbg_log(x)
                y = fp8_fused_prefill_linear(
                    x, layer._firefly_w_fp8, layer._firefly_s_inv, layer._firefly_c_n
                )
                if bias is not None:
                    y = y + bias
                return y
            if getattr(layer, "_firefly_unfused", False):
                cuda_dequant = getattr(layer, "_firefly_unfused_cuda_dequant", False)
                _ff_unfused_dbg_log(x, cuda_dequant)
                if cuda_dequant:
                    # CUDA 反量化(fast, 与 B-load 逐 bit 一致)
                    w_int8 = _load_fused_mod().fp8_dequant_only(
                        layer._firefly_w_fp8, layer._firefly_s_inv, layer._firefly_c_n
                    )
                else:
                    # PyTorch 反量化(回退, 正确但慢)
                    w_int8, _ = fp8_to_int8(layer._firefly_w_fp8, layer._firefly_s_inv)
                y = int8_prefill_linear(x, w_int8, layer._firefly_c_n)
                if bias is not None:
                    y = y + bias
                return y

        # 小 M(decode) 或 firefly 禁用 → 上游 fp8 Marlin
        if self.block_quant:
            weight_scale = layer.weight_scale_inv
        else:
            weight_scale = layer.weight_scale
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            input_dtype=self.marlin_input_dtype,
            bias=bias,
        )

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass
