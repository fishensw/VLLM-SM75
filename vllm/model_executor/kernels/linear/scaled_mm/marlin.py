# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 overlay: MarlinFP8ScaledMMLinearKernel + firefly(fp8→int8 prefill)混合。

全文件拷贝自上游 v0.28.0, 仅新增 firefly 分支(env 门控, off=上游行为):
  - process_weights_after_loading: repack 前快照干净 fp8 权重 [N,K] + block scale
    [N/128,K/128], 现反量化成 per-channel int8 缓存(B1-easy, 小模型够用)。
  - apply_weights: 大 M(prefill)走 int8 cutlass(IMMA), 小 M(decode)走上游
    weight-only fp8 Marlin(带宽最优, fp8=int8=1B/参数 换 int8 无带宽收益)。

fp8 对称无 zp, 反量化比 AWQ 非对称 int4 简单; int8 GEMM 复用 int8_prefill_linear。
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
_firefly_int8_logged = False


def _ff_dbg_log(x: torch.Tensor, take_int8: bool) -> None:
    # 首次 int8 prefill 触发时打一次点(确认机制生效, 不刷屏)。
    global _firefly_int8_logged
    if take_int8 and not _firefly_int8_logged:
        _firefly_int8_logged = True
        logger.info(
            "[firefly-fp8] int8 IMMA prefill active (first M=%d)",
            x.numel() // x.shape[-1],
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
        # firefly-fp8: repack 前快照干净 fp8 权重 [N,K] + block scale [N/128,K/128],
        # 现反量化成 per-channel int8 并缓存(B1-easy, 小模型 load 一次开销可接受)。
        # CUDA 上 process_fp8_weight_block_strategy 只 pad 权重、scale 原样返回,
        # 故此处快照即干净 canonical 布局。
        if self._firefly_enabled():
            w_fp8 = layer.weight.contiguous()  # [N, K] fp8 e4m3
            s_inv = layer.weight_scale_inv.contiguous()  # [N/128, K/128]
            w_int8, c_n = fp8_to_int8(w_fp8, s_inv)
            layer._firefly_w_int8 = w_int8  # [N, K] int8 行主序
            layer._firefly_c_n = c_n  # [N] fp32

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
        # firefly-fp8: 大 M(prefill)把 fp8 现反量化缓存的 int8 走 cutlass(IMMA);
        # 小 M(decode)走上游 weight-only fp8 Marlin(带宽最优)。
        # int8 GEMM 复用 int8_prefill_linear(per-token 动态 int8 激活)。
        w_int8 = getattr(layer, "_firefly_w_int8", None)
        take_int8 = w_int8 is not None and self._firefly_m_large(x)
        _ff_dbg_log(x, take_int8)
        if take_int8:
            y = int8_prefill_linear(x, w_int8, layer._firefly_c_n)
            if bias is not None:
                y = y + bias
            return y

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
