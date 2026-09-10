# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 overlay: MarlinLinearKernel + firefly(int4→int8 prefill)混合。

全文件拷贝自上游 v0.28.0, 仅新增 firefly 分支(env 门控, off=上游行为):
  - process_weights_after_loading: repack 前快照干净 int4 布局 [N, K/8] + scale
  - apply_weights: 大 M(prefill)走 int8 cutlass(IMMA), 小 M(decode)走上游 int4 Marlin

见 vllm/model_executor/layers/quantization/utils/firefly.py。
"""

import logging

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
    dequant_marlin_to_int8,
    dequant_marlin_to_int8_cached,
    firefly_active_int4,
    int8_prefill_linear,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = logging.getLogger(__name__)

_firefly_int4_fallback_logged = False


def _ff_int4_fallback_log() -> None:
    # act-order int4 模型 firefly 回退上游 Marlin 时打一次明显日志(不刷屏):
    # B1-hard repack 反推仅针对 has_perm=false, act-order(g_idx 排序)不支持。
    global _firefly_int4_fallback_logged
    if _firefly_int4_fallback_logged:
        return
    _firefly_int4_fallback_logged = True
    logger.warning(
        "[firefly-int4] 回退上游 Marlin: 模型为 act-order(g_idx 排序), "
        "B1-hard 反推仅支持非 act-order, firefly prefill 对该模型禁用"
    )


# ---- firefly 混合 op(dynamo/inductor 兼容) ----
# firefly 分支判据(`m > min_m`, m 是动态 shape) 若直接留在编译图里, inductor 会把
# firefly 路径(含 custom op)一起编进 decode(M=1) 的图, 实测 decode 从 27 tok/s 掉到
# 4.8 tok/s(baseline 无 firefly 时该分支被静态消除)。
# 图断点方案走不通: torch._dynamo.disable + fullgraph=False 与 VllmBackend
# ("can only be called once", 只收单图)和 AOT compile("不支持图断点")冲突。
# 故把"门控 + firefly/Marlin 选择 + 执行"整体包成一个 torch.library custom op:
# 图里只剩一个不透明 leaf, inductor 看不到分支, decode 图干净(=baseline 性能)。
# op 内部按真实 M 求值: prefill(m>min_m)现反量化 int8 走 IMMA, decode 走上游 Marlin。
@torch.library.custom_op("firefly::hybrid_linear", mutates_args=())
def _ff_hybrid_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s_marlin: torch.Tensor,
    w_zp_marlin: torch.Tensor,
    w_gidx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    input_global_scale: torch.Tensor,
    bias: torch.Tensor,
    w_s_clean: torch.Tensor,
    w_zp_clean: torch.Tensor,
    c_n: torch.Tensor,
    size_k: int,
    size_n: int,
    padded_k: int,
    padded_n: int,
    gs: int,
    min_m: int,
    has_zp: bool,
    is_k_full: bool,
    use_recip: bool,
) -> torch.Tensor:
    m = x.numel() // x.shape[-1]
    if m > min_m:
        # prefill: transient 反量化 int4→int8 + cutlass_scaled_mm(SM75 即 IMMA)
        w_int8 = dequant_marlin_to_int8_cached(
            w_q,
            w_s_clean,
            c_n,
            gs,
            size_k,
            size_n,
            padded_k,
            padded_n,
            w_zp=(w_zp_clean if w_zp_clean.numel() > 0 else None),
            use_recip=use_recip,
        )
        return int8_prefill_linear(x, w_int8, c_n)
    # decode: 上游 int4 Marlin
    wtype = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    return apply_gptq_marlin_linear(
        input=x,
        weight=w_q,
        weight_scale=w_s_marlin,
        weight_zp=w_zp_marlin,
        g_idx=w_gidx,
        g_idx_sort_indices=g_idx_sort_indices,
        workspace=workspace,
        wtype=wtype,
        input_size_per_partition=size_k,
        output_size_per_partition=size_n,
        is_k_full=is_k_full,
        input_global_scale=(
            input_global_scale if input_global_scale.numel() > 0 else None
        ),
        bias=bias if bias.numel() > 0 else None,
        input_dtype=x.dtype,
    )


@_ff_hybrid_linear.register_fake
def _(
    x,
    _w_q,
    _w_s_marlin,
    _w_zp_marlin,
    _w_gidx,
    _g_idx_sort_indices,
    _workspace,
    _input_global_scale,
    _bias,
    _w_s_clean,
    _w_zp_clean,
    _c_n,
    _size_k,
    size_n,
    _padded_k,
    _padded_n,
    _gs,
    _min_m,
    _has_zp,
    _is_k_full,
    _use_recip,
):
    return torch.empty(x.shape[:-1] + (size_n,), dtype=x.dtype, device=x.device)


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
        # 只有 hard 模式: 不常驻副本, prefill 现从 marlin 布局反。B1-hard 反推
        # 仅针对 has_perm=false, act-order(g_idx 排序)不支持 → 禁用(回退上游)。
        return (
            firefly_active_int4()
            and self.config.weight_type in (scalar_types.uint4b8, scalar_types.uint4)
            and self.config.act_type in (torch.float16, torch.bfloat16)
            and not self.config.has_g_idx
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

        # firefly 回退: act-order int4 模型 B1-hard 不支持, firefly 禁用 → 上游 Marlin。
        if (
            firefly_active_int4()
            and c.weight_type in (scalar_types.uint4b8, scalar_types.uint4)
            and c.act_type in (torch.float16, torch.bfloat16)
            and c.has_g_idx
        ):
            _ff_int4_fallback_log()

        # hybrid: repack 前快照(此时 weight/scale 仍为干净布局, 未 permute/repack)。
        # clone 防后续 in-place 修改。scale 统一转置成 N 在前 [N, K/gs]
        # (compressed-tensors 已是 N 在前; AWQ 是 K 在前 [K/gs, N] 需转置)。
        # 只有 hard 模式: 存干净 scale(小, ~0 字节/参数) + 维度; 非对称(AWQ)另存
        # 干净 qzeros(packed, 小, dequant 时解包); 不存 int4 副本, prefill 步现从
        # marlin 布局反回(单权重, 不爆显存)。
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
            # 只有 hard: 不存干净 int4 副本, prefill 步现从 marlin 布局反回。

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

        # firefly c_n 缓存: load 时算一次 per-channel c_n(纯权重导出, M/chunk 无关),
        # 运行时单遍反量化复用(免两遍 amax, 实测省反量化 ~46%)。放在 repack
        # (transform_w_q) 之后: 此时 w_q 已是 marlin 布局, 与 apply_weights 读取一致。
        # 复用现有全量反量化取 c_n(w_int8 临时算出即弃)。
        if self._firefly_enabled():
            # 只有 hard: c_n 从 marlin 布局现算(无干净 int4 副本)
            _, layer._firefly_c_n = dequant_marlin_to_int8(
                getattr(layer, self.w_q_name).data,
                layer._firefly_ws,
                layer._firefly_gs,
                layer._firefly_size_k,
                layer._firefly_size_n,
                layer._firefly_padded_k,
                layer._firefly_padded_n,
                w_zp=layer._firefly_wzp,
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hybrid: 大 M(prefill)把 int4 现反量化成 int8 走 cutlass(IMMA), 小 M(decode)
        # 走上游 int4 Marlin。整个"门控+选择+执行"包在 _ff_hybrid_linear(op) 里,
        # 对 inductor 不透明(图里一个 leaf), 避免 m>min_m 数据依赖分支拖慢 decode。
        # 只有 hard: 无干净 int4 副本, op 内走 marlin 布局反量化(单权重)。
        # input_global_scale/bias/w_zp_clean 用空张量表示 None(custom op 不收 None)。
        if self._firefly_enabled():
            c = self.config
            w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)
            empty = torch.empty(0, device=x.device)
            igs = getattr(layer, "input_global_scale", None)
            igs_t = igs.data if igs is not None else empty
            bias_t = bias.data if bias is not None else empty
            wzp_c = getattr(layer, "_firefly_wzp", None)
            wzp_c_t = wzp_c if wzp_c is not None else empty
            return _ff_hybrid_linear(
                x,
                w_q,
                w_s,
                w_zp,
                w_gidx,
                layer.g_idx_sort_indices,
                self.workspace,
                igs_t,
                bias_t,
                layer._firefly_ws,
                wzp_c_t,
                layer._firefly_c_n,
                c.partition_weight_shape[0],
                c.partition_weight_shape[1],
                layer._firefly_padded_k,
                layer._firefly_padded_n,
                layer._firefly_gs,
                envs.VLLM_FIREFLY_MIN_M,
                c.zero_points,
                self.is_k_full,
                envs.VLLM_FIREFLY_DEQUANT_MODEL == "fast",
            )

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
