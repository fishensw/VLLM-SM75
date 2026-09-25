# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int8 权重 (compressed-tensors int-quantized) 的 firefly 接入 scheme (SM75)。

覆盖两种激活形态 (权重侧完全相同: int8 [N,K] 行主序 + per-channel scale [N],
即 firefly 的 c_n, 与 w_int8 = round(w / c_n)、c_n = amax_k/127 约定一致):
- W8A16: 无激活量化 (input_activations=None);
- W8A8:  动态 per-token int8 激活 (SmoothQuant, dynamic=true, 无静态 scale 张量)。
两者前向都走 firefly 的 int8_prefill_linear (per-token 动态 int8 激活 +
cutlass_scaled_mm, 见 utils/firefly.py) —— 该 GEMM 本就是对激活做 per-token 动态
int8 量化, W8A8 的 dynamic 激活语义与它完全对口, 权重布局相同, 故同一 scheme 通吃。

stock vllm 0.29.0 对 W8A16 (int-quantized int8, 无 input_activations) 在
CompressedTensorsConfig._get_scheme_from_parts 里不匹配任何子分支, 落到末尾
raise NotImplementedError, 故需本 scheme 接管。

接入方式: maybe_apply() monkey-patch CompressedTensorsConfig.get_scheme —— 命中
上述 int8 权重条件时返回 FireflyInt8Scheme, 其余情况原样调原 get_scheme。只在
firefly 总开关 VLLM_FIREFLY 开启 (1/auto) 时打 patch; 关则完全不碰上游 (行为不变)。
幂等: 原函数打标记属性 ._sm75_int8_patched, 二次 import 不重复 patch。
"""

import logging
from collections.abc import Callable
from functools import wraps

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)

logger = logging.getLogger(__name__)

__all__ = ["FireflyInt8Scheme", "maybe_apply"]

# firefly 总开关的幂等 patch 标记 (打到被替换的 get_scheme 函数上)。
_PATCHED_FLAG = "_sm75_int8_patched"
# int-quantized int8 权重会走的 compressed-tensors format 取值。
_INT8_WEIGHT_FORMATS = ("int-quantized", "naive-quantized")


def _firefly_on() -> bool:
    """firefly 总开关是否开: 优先复用 vllm.envs_sm75._firefly_mode (与上游
    VLLM_FIREFLY 语义一致: 未设/0=关, 1/auto/on/true/yes=开); 该模块不可用时
    回退直接读环境变量 (归一化规则相同)。"""
    try:
        from vllm.envs_sm75 import _firefly_mode

        return _firefly_mode() == "1"
    except Exception:  # noqa: BLE001 - 无该模块/无 CUDA 环境时回退直读 env
        import os

        return os.getenv("VLLM_FIREFLY", "").strip().lower() in (
            "1",
            "auto",
            "on",
            "true",
            "yes",
        )


class FireflyInt8Scheme(CompressedTensorsScheme):
    """int8 权重 (compressed-tensors int-quantized int8, per-channel) 走 firefly
    int8 GEMM 的 linear scheme; 兼容 W8A16 (无激活量化) 与 W8A8 (动态 per-token
    int8 激活, SmoothQuant) —— 权重布局相同, 前向都是 per-token 动态 int8 GEMM。

    对齐 compressed_tensors 的 scheme 协议: 实现 CompressedTensorsScheme 的
    全部抽象方法 (get_min_capability / create_weights / apply_weights /
    process_weights_after_loading)。参数命名与上游 W8A8Int8 对齐 (weight /
    weight_scale), **不做 pack** —— 权重保留非打包 int8 [N,K] 行主序 (与
    checkpoint 的 weight 张量一一对应)。
    """

    @classmethod
    def get_min_capability(cls) -> int:
        # Turing (sm75) 及以上; 与 WNA16 一致。
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ) -> None:
        """分配非打包 int8 参数 (W8A16/W8A8 权重侧相同)。命名/入参与上游
        W8A8Int8 对齐: weight [output, input] int8 行主序 + per-channel
        weight_scale [output, 1] float32。
        """
        output_size_per_partition = sum(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.output_partition_sizes = output_partition_sizes
        layer.params_dtype = params_dtype
        if not hasattr(layer, "has_bias"):
            layer.has_bias = False

        # 非打包 int8 权重 [N, K] 行主序 (N = output, K = input)。
        weight = ModelWeightParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
        )

        # per-channel scale [N,1] (checkpoint 的 weight_scale); float32 (对齐上游
        # W8A8Int8 的 channel 分支 scale dtype)。
        weight_scale = ChannelQuantScaleParameter(
            output_dim=0,
            weight_loader=weight_loader,
            data=torch.empty(
                output_size_per_partition,
                1,
                dtype=torch.float32,
            ),
        )

        # 注册名 "weight": int-quantized (int8 非打包) format 的标准键名, W8A16 与
        # W8A8 权重侧布局相同(区别只在激活侧), checkpoint 键名都是 weight —— 故同一
        # scheme 通吃两者。不要误用 WNA16(int4 打包)的 weight_packed 键名。
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """把加载好的 int8 权重 / scale 整理成 firefly int8 GEMM 的入参。

        ff_int8_w_int8 [N,K] 行主序 int8 (w_int8); ff_int8_c_n [N] float32
        (c_n = amax_k/127, 与 firefly.py 的 c_n 约定一致)。w_int8 若已
        contiguous 则直接引用 (不 copy, 省显存); scale 兜底 [N,1]/[N] reshape(-1)
        转 float32。处理完释放原 weight / weight_scale 参数 (数据由新属性持有,
        无重复显存)。
        """
        w = layer.weight.data
        if w.dtype != torch.int8:
            w = w.to(torch.int8)
        # 已是 contiguous 直接引用, 不 copy (省显存)。
        layer.ff_int8_w_int8 = w.contiguous()
        layer.ff_int8_c_n = (
            layer.weight_scale.data.reshape(-1).float().contiguous()
        )
        # 释放原占位参数 (数据已转移到 ff_int8_* 属性, 无重复显存)。
        try:
            del layer.weight
        except Exception:  # noqa: BLE001 - 删参兜底, 不影响正确性
            pass
        try:
            del layer.weight_scale
        except Exception:  # noqa: BLE001
            pass

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """firefly int8 GEMM: per-token 动态 int8 激活 + cutlass_scaled_mm。
        延迟 import firefly 模块 (避免与 compressed_tensors 的循环依赖); bias
        非 None 时加回 (Qwen 等 proj 均无 bias)。"""
        from vllm.model_executor.layers.quantization.utils.firefly import (
            int8_prefill_linear,
        )

        out = int8_prefill_linear(
            x, layer.ff_int8_w_int8, layer.ff_int8_c_n
        )
        if bias is not None:
            out = out + bias
        return out


def _is_sm75_int8(
    weight_quant, input_quant: "object | None", format: "str | None"
) -> bool:
    """是否本模块要接管的 int8 权重层 (W8A16 或 W8A8)。

    权重侧条件 (两者相同): int 型 int8 权重 + channel/group + 对称 + 非 dynamic +
    int-quantized/naive-quantized format。
    激活侧: 放行 input_quant 为 None (W8A16) 或 int 型动态 (W8A8 / SmoothQuant
    per-token int8, dynamic=true, 无静态 scale 张量) —— firefly 的 GEMM 本就对激活
    做 per-token 动态 int8 量化, 两种都对口。W8A8 静态激活 (有 scale 张量) 或 fp8
    激活不接管, 回落上游。ignore 列表 (BF16 层) 由上游 get_scheme_dict 返 None 提前
    回落 UnquantizedLinearMethod, 不会走到这里。
    """
    if weight_quant is None:
        return False
    if getattr(weight_quant, "type", None) != "int":
        return False
    if getattr(weight_quant, "num_bits", None) != 8:
        return False
    if getattr(weight_quant, "strategy", None) not in ("channel", "group"):
        return False
    if getattr(weight_quant, "symmetric", False) is not True:
        return False
    if getattr(weight_quant, "dynamic", False):
        return False
    if format is not None and format not in _INT8_WEIGHT_FORMATS:
        return False
    # 激活: None (W8A16) 或 int 型 (W8A8 动态 per-token int8)。
    if input_quant is not None:
        if getattr(input_quant, "type", None) != "int":
            return False
        if getattr(input_quant, "num_bits", None) != 8:
            return False
    return True


def maybe_apply() -> None:
    """firefly 开启时 monkey-patch CompressedTensorsConfig.get_scheme。

    命中的 int8 权重层 (W8A16/W8A8) 返回 FireflyInt8Scheme, 其余原样调原
    get_scheme。仅在
    VLLM_FIREFLY 开启时 patch, 幂等 (防重复 import 二次 patch); 失败静默
    (不破坏 compressed-tensors 主链路)。由 compressed_tensors.py 末尾的
    overlay 注入调用。
    """
    if not _firefly_on():
        return

    try:
        from vllm.model_executor.layers.quantization.compressed_tensors import (
            compressed_tensors as _ct_mod,
        )

        cfg_cls = _ct_mod.CompressedTensorsConfig
        # get_scheme 是 CompressedTensorsConfig 的实例方法 (未 @classmethod);
        # 从类 dict 取底层函数。幂等: 若它已带 _PATCHED_FLAG (上一次 import
        # 已 patch) 则跳过; 首次则它指向上游原函数, 无 flag。
        _orig_get_scheme = cfg_cls.__dict__.get("get_scheme")
        if _orig_get_scheme is None or getattr(
            _orig_get_scheme, _PATCHED_FLAG, False
        ):
            return

        @wraps(_orig_get_scheme)
        def _patched_get_scheme(
            self, layer: torch.nn.Module, layer_name: str | None = None
        ):
            scheme_dict = self.get_scheme_dict(layer, layer_name)
            weight_quant = scheme_dict.get("weights") if scheme_dict else None
            input_quant = scheme_dict.get("input_activations") if scheme_dict else None
            fmt = scheme_dict.get("format") if scheme_dict else None
            if _is_sm75_int8(weight_quant, input_quant, fmt):
                # FireflyInt8Scheme 与本模块同文件已定义, 直接引用 (免 self-import)。
                scheme = FireflyInt8Scheme()
                # 对齐上游 get_scheme 尾部: 能力门 + 选 scheme 的 debug 日志。
                self._check_scheme_supported(scheme.get_min_capability())
                logger.debug(
                    "Using scheme: %s for %s",
                    scheme.__class__.__name__,
                    layer_name,
                )
                return scheme
            # 其余情况原样走原 get_scheme (含 ignore / 其它 scheme)。
            return _orig_get_scheme(self, layer, layer_name)

        setattr(_patched_get_scheme, _PATCHED_FLAG, True)
        cfg_cls.get_scheme = _patched_get_scheme
        logger.info("SM75 firefly int8: patched CompressedTensorsConfig.get_scheme")
    except Exception as e:  # noqa: BLE001 - patch 失败静默, 不破坏主链路
        logger.warning("SM75 firefly int8 maybe_apply failed: %s", e)
