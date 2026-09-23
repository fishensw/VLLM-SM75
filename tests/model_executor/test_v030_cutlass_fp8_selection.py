"""Regression coverage for the CUTLASS FP8 capability gate (upstream #53376)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "vllm/model_executor/kernels/linear/scaled_mm/cutlass.py"


def load_gate(platform, probe):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CutlassFP8ScaledMMLinearKernel")
    gate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "is_supported")
    gate.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[gate], type_ignores=[]))
    namespace = {"current_platform": platform, "ops": SimpleNamespace(cutlass_scaled_mm_supports_fp8=probe)}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["is_supported"]


@pytest.mark.parametrize("capability,expected", [(75, False), (80, False), (86, False), (89, True), (90, True), (120, True)])
def test_explicit_capability_uses_build_probe(capability, expected):
    probe = Mock(side_effect=lambda cap: cap >= 89)
    gate = load_gate(SimpleNamespace(is_cuda=lambda: True), probe)
    supported, reason = gate(None, capability)
    assert supported is expected
    probe.assert_called_once_with(capability)
    if expected:
        assert reason is None
    else:
        assert "CUTLASS FP8" in reason
        assert str(capability) in reason


@pytest.mark.parametrize("capability", [75, 90, None])
def test_omitted_capability_queries_actual_device(capability):
    probe = Mock(side_effect=lambda cap: cap >= 89)
    device = None if capability is None else SimpleNamespace(to_int=lambda: capability)
    platform = SimpleNamespace(is_cuda=lambda: True, get_device_capability=lambda: device)
    supported, _ = load_gate(platform, probe)(None)
    probe.assert_called_once_with(-1 if capability is None else capability)
    assert supported is (capability == 90)


def test_cuda_build_can_reject_nominally_supported_hardware():
    probe = Mock(return_value=False)
    supported, reason = load_gate(SimpleNamespace(is_cuda=lambda: True), probe)(None, 89)
    assert supported is False
    assert "CUDA build" in reason


def test_non_cuda_does_not_call_cuda_probe():
    probe = Mock(side_effect=AssertionError("must not probe CUDA"))
    assert load_gate(SimpleNamespace(is_cuda=lambda: False), probe)(None, 90) == (False, "requires CUDA.")
    probe.assert_not_called()


@pytest.mark.parametrize("capability,expected", [(75, "marlin"), (80, "marlin"), (86, "marlin"), (90, "cutlass")])
def test_runtime_selector_falls_through_to_marlin(monkeypatch, capability, expected):
    # This integration assertion runs in the image; the isolated overlay can
    # still execute all gate tests on hosts without PyTorch or CUDA.
    torch = pytest.importorskip("torch")
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_FP8_KERNELS,
        CutlassFP8ScaledMMLinearKernel,
        FP8ScaledMMLinearLayerConfig,
        MarlinFP8ScaledMMLinearKernel,
        choose_scaled_mm_linear_kernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm import cutlass
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym
    from vllm.platforms import PlatformEnum, current_platform

    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(cutlass.ops, "cutlass_scaled_mm_supports_fp8", lambda cap: cap >= 89)
    monkeypatch.setattr(current_platform, "has_device_capability", lambda required, device_id=0: capability >= (required if isinstance(required, int) else required[0] * 10 + required[1]))
    monkeypatch.setattr(current_platform, "is_device_capability_family", lambda family, device_id=0: capability // 10 == family // 10)
    monkeypatch.setattr(current_platform, "supports_fp8", lambda: capability >= 89)
    config = FP8ScaledMMLinearLayerConfig(
        activation_quant_key=kFp8StaticTensorSym, weight_quant_key=kFp8StaticTensorSym,
        weight_shape=(2048, 2048), input_dtype=torch.float16, out_dtype=torch.float16,
    )
    chosen = choose_scaled_mm_linear_kernel(config, _POSSIBLE_FP8_KERNELS, compute_capability=capability, quantization="fp8_w8a8")
    assert chosen is (MarlinFP8ScaledMMLinearKernel if expected == "marlin" else CutlassFP8ScaledMMLinearKernel)
