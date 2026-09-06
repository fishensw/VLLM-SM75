# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 overlay: MarlinLinearKernel + firefly(int4→int8 prefill)混合。

全文件拷贝自上游 v0.28.0, 仅新增 firefly 分支(env 门控, off=上游行为):
  - process_weights_after_loading: repack 前快照干净 int4 布局 [N, K/8] + scale
  - apply_weights: 大 M(prefill)走 int8 cutlass(IMMA), 小 M(decode)走上游 int4 Marlin

见 vllm/model_executor/layers/quantization/utils/firefly.py。
"""

import torch

from vllm import _custom_ops as ops
import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_act_int8_process_scales,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_pad_dim,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_sort_g_idx,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from vllm.model_executor.layers.quantization.utils.firefly import (
    dequant_clean_int4_to_int8,
    dequant_marlin_to_int8,
    int8_prefill_linear,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class MarlinLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Marlin uses inline PTX, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "Marlin only supported on CUDA"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by"
                f"  Marlin, supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "Marlin, supported group sizes are: "
                f"{MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        if c.has_g_idx:
            # Act-order couples K to the full-model group layout, so tile
            # padding is not supported; keep the strict shape check.
            return check_marlin_supports_shape(
                c.partition_weight_shape[1],  # out_features
                c.partition_weight_shape[0],  # in_features
                c.full_weight_shape[0],  # in_features
                c.group_size,
            )

        # A group straddling TP ranks cannot be fixed by padding.
        if (
            c.group_size != -1
            and c.group_size < c.full_weight_shape[0]
            and c.partition_weight_shape[0] % c.group_size != 0
        ):
            return False, (
                f"in_features per partition {c.partition_weight_shape[0]} is "
                f"not divisible by group_size = {c.group_size}."
            )

        # Tile misalignment is fixed by zero-padding at weight prep.
        return True, None

    # ---- firefly 门控(int4 权重 → int8 prefill) ----
    def _firefly_enabled(self) -> bool:
        # 仅对 int4 权重 + fp16/bf16 激活启用; int8/fp8 激活(W4A8/FP8)走上游,
        # 不参与 firefly 反量化。
        # gate 卡 scalar type 而非量化框架:
        #   uint4b8 = 对称 int4(zp=8, compressed-tensors/GPTQ-sym)
        #   uint4   = 非对称 int4(per-group qzeros, AWQ)
        # 反量化统一 w_deq=(q-zp)*s, 对称 zp=8 是其特例(不回归)。
        return (
            envs.VLLM_FIREFLY
            and self.config.weight_type in (scalar_types.uint4b8, scalar_types.uint4)
            and self.config.act_type in (torch.float16, torch.bfloat16)
        )

    def _firefly_m_large(self, x: torch.Tensor) -> bool:
        m = x.numel() // x.shape[-1]
        return m > envs.VLLM_FIREFLY_MIN_M

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "W8A8 is not supported by marlin kernel."
            )

        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data * 512
            )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        size_k, size_n = c.partition_weight_shape
        if c.has_g_idx:
            # Act-order shapes were strictly validated in can_implement.
            padded_n, padded_k = size_n, size_k
        else:
            padded_n, padded_k = marlin_padded_nk(size_n, size_k, c.group_size)

        # hybrid: repack 前快照(此时 weight/scale 仍为干净布局, 未 permute/repack)。
        # clone 防后续 in-place 修改。scale/wp 统一转置成 N 在前 [N, K/gs]/[N, K/8]
        # (compressed-tensors 已是 N 在前; AWQ 是 K 在前 [K/gs, N]/[K/8, N] 需转置)。
        # 两模式都存干净 scale(小, ~0 字节/参数) + 维度; 非对称(AWQ)另存干净
        # qzeros(packed, 小, dequant 时解包); B1-easy 另存干净 int4 副本
        # (0.5 字节/参数, 大模型 OOM), B1-hard 不存, prefill 步现从 marlin 布局反回。
        if self._firefly_enabled():
            ws = getattr(layer, self.w_s_name).data.clone()
            if ws.shape[0] != size_n:  # AWQ K 在前 → N 在前
                ws = ws.t().contiguous()
            layer._firefly_ws = ws  # [N, K/gs]
            layer._firefly_gs = (
                self.config.group_size if self.config.group_size > 0 else -1
            )
            layer._firefly_size_k = size_k
            layer._firefly_size_n = size_n
            layer._firefly_padded_k = padded_k
            layer._firefly_padded_n = padded_n
            # 非对称(AWQ): 快照干净 qzeros(packed [N/8, K/gs]); 对称: None(zp=8)。
            layer._firefly_wzp = (
                getattr(layer, self.w_zp_name).data.clone() if c.zero_points else None
            )
            # B1-hard 仅支持非 act-order(repack 反转推导针对 has_perm=false);
            # act-order 回退 B1-easy(存干净 int4 副本)。
            use_easy = (
                envs.VLLM_FIREFLY_MODE == "easy" or c.has_g_idx
            )
            if use_easy:
                wp = getattr(layer, self.w_q_name).data.clone()
                if wp.shape[0] != size_n:  # AWQ K 在前 → N 在前
                    wp = wp.t().contiguous()
                layer._firefly_wp = wp  # [N, K/8]

        # Allocate marlin workspace, reusing existing storage on reload.
        self.workspace = marlin_make_workspace_new(
            device, existing=getattr(self, "workspace", None)
        )

        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.gptq_marlin_repack(
                marlin_pad_qweight(
                    x.data.contiguous(), size_n, size_k, padded_n, padded_k
                ),
                perm=layer.g_idx_sort_indices,
                size_k=padded_k,
                size_n=padded_n,
                num_bits=c.weight_type.size_bits,
                is_a_8bit=is_a_8bit,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                marlin_pad_scales(
                    x.data.contiguous(),
                    size_n,
                    size_k,
                    padded_n,
                    padded_k,
                    c.group_size,
                ),
                size_k=padded_k,
                size_n=padded_n,
                group_size=c.group_size,
                is_a_8bit=is_a_8bit,
            )

            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            if c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                layer.register_parameter(
                    "input_global_scale",
                    torch.nn.Parameter(input_global_scale, requires_grad=False),
                )
            else:
                layer.input_global_scale = None
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            replace_parameter(
                layer, "g_idx_sort_indices", g_idx_sort_indices, prefer_copy=True
            )
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = size_k // c.group_size if c.group_size != -1 else 1
            padded_grouped_k = padded_k // c.group_size if c.group_size != -1 else 1
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    marlin_pad_scales(
                        unpack_cols(
                            x.t(),
                            c.weight_type.size_bits,
                            grouped_k,
                            size_n,
                        ),
                        size_n,
                        size_k,
                        padded_n,
                        padded_k,
                        c.group_size,
                    ),
                    size_k=padded_grouped_k,
                    size_n=padded_n,
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data = marlin_permute_bias(
                marlin_pad_dim(layer.bias, size_n, padded_n)
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hybrid: 大 M(prefill)把 int4 现反量化成 int8 走 cutlass(IMMA)。
        # 不缓存 int8, 每次 prefill 步重算。小 M(decode)走上游 int4 Marlin, 不吃反量化开销。
        # B1-easy(有 _firefly_wp): 从干净 int4 副本反量化。
        # B1-hard(无副本): 从 marlin 布局现反回干净 int4(只存 0.53 字节/参数)。
        # w_zp: 对称 None(zp=8); 非对称(AWQ) 干净 qzeros, dequant 内 (q-zp)*s。
        if getattr(layer, "_firefly_ws", None) is not None and self._firefly_m_large(
            x
        ):
            hybrid_wzp = getattr(layer, "_firefly_wzp", None)
            if getattr(layer, "_firefly_wp", None) is not None:
                w_int8, c_n = dequant_clean_int4_to_int8(
                    layer._firefly_wp,
                    layer._firefly_ws,
                    layer._firefly_gs,
                    w_zp=hybrid_wzp,
                )
            else:
                w_int8, c_n = dequant_marlin_to_int8(
                    getattr(layer, self.w_q_name).data,
                    layer._firefly_ws,
                    layer._firefly_gs,
                    layer._firefly_size_k,
                    layer._firefly_size_n,
                    layer._firefly_padded_k,
                    layer._firefly_padded_n,
                    w_zp=hybrid_wzp,
                )
            return int8_prefill_linear(x, w_int8, c_n)

        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin

        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=self.is_k_full,
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )
