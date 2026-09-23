"""Scope the Humming SM75 FP16 overflow gate and verify actual selection."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "vllm/model_executor/kernels/linear/scaled_mm/humming.py"


def load_gate(capability, *, cuda=True, kernel="HummingFP8ScaledMMLinearKernel"):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == kernel)
    gate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "can_implement")
    gate.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[gate], type_ignores=[]))
    namespace = {
        "current_platform": SimpleNamespace(
            is_cuda=lambda: cuda,
            is_device_capability=lambda required: required == capability,
        ),
        "torch": SimpleNamespace(float16="float16"),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["can_implement"]


def config(input_dtype="float16", out_dtype="float16", scale="tensor"):
    return SimpleNamespace(
        input_dtype=input_dtype, out_dtype=out_dtype,
        weight_quant_key=SimpleNamespace(scale=SimpleNamespace(group_shape=SimpleNamespace(
            is_per_tensor=lambda: scale == "tensor",
            is_per_channel=lambda: scale == "channel",
        ))),
    )


@pytest.mark.parametrize("scale", ["tensor", "channel"])
def test_sm75_fp16_tensor_and_expanded_channel_are_rejected(scale):
    allowed, reason = load_gate(75)(None, config(scale=scale))
    assert not allowed
    assert "overflow before scaling" in reason and "Marlin" in reason


@pytest.mark.parametrize("capability", [70, 80, 86, 89, 90, 120])
def test_other_devices_keep_existing_eligibility(capability):
    assert load_gate(capability)(None, config()) == (True, None)


@pytest.mark.parametrize("input_dtype,out_dtype,scale", [
    ("float8_e4m3fn", "float16", "tensor"),
    ("float8_e4m3fn", "float16", "channel"),
    ("bfloat16", "bfloat16", "tensor"),
    ("float16", "bfloat16", "tensor"),
    ("float16", "float16", "group"),
    ("float16", "float16", "block"),
])
def test_unaffected_dtypes_and_group_scales(input_dtype, out_dtype, scale):
    assert load_gate(75)(None, config(input_dtype, out_dtype, scale)) == (True, None)


def test_other_platform_and_int8_gate_are_unchanged():
    assert load_gate(75, cuda=False)(None, config()) == (True, None)
    assert load_gate(75, kernel="HummingInt8ScaledMMLinearKernel")(None, object()) == (True, None)


@pytest.mark.parametrize("scale", ["tensor", "channel"])
@pytest.mark.parametrize("static_activation", [False, True])
@pytest.mark.parametrize("backend,capability,expected", [
    ("auto", 75, "MarlinFP8ScaledMMLinearKernel"),
    ("humming", 75, "reject"),
    ("auto", 80, "HummingFP8ScaledMMLinearKernel"),
])
def test_runtime_selection_and_explicit_backend(monkeypatch, backend, capability, expected, scale, static_activation):
    torch = pytest.importorskip("torch")
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import (
        FP8ScaledMMLinearLayerConfig, _POSSIBLE_WFP8A16_KERNELS,
        choose_scaled_mm_linear_kernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm import humming
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTensorSym, kFp8StaticTensorSym, kFp8StaticChannelSym,
    )
    from vllm.platforms import PlatformEnum, current_platform

    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(current_platform, "is_device_capability", lambda required, device_id=0: required == capability)
    monkeypatch.setattr(current_platform, "has_device_capability", lambda required, device_id=0: capability >= (required if isinstance(required, int) else required[0] * 10 + required[1]))
    monkeypatch.setattr(humming, "has_humming", lambda: True)
    layer = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticTensorSym if scale == "tensor" else kFp8StaticChannelSym,
        # CT W8A16 keeps this key although Humming consumes actual FP16 input.
        activation_quant_key=kFp8StaticTensorSym if static_activation else kFp8DynamicTensorSym,
        weight_shape=(256, 256), input_dtype=torch.float16, out_dtype=torch.float16,
    )
    # Selection is mocked; CPU-only image builds must not auto-detect a GPU.
    runtime = VllmConfig(device_config=DeviceConfig(device="cuda"))
    runtime.kernel_config.linear_backend = backend
    with set_current_vllm_config(runtime):
        def choose():
            return choose_scaled_mm_linear_kernel(
                layer, _POSSIBLE_WFP8A16_KERNELS, compute_capability=capability,
                quantization="w8a16_fp8",
            )
        if expected == "reject":
            with pytest.raises(ValueError, match="overflow before scaling"):
                choose()
        else:
            assert choose().__name__ == expected