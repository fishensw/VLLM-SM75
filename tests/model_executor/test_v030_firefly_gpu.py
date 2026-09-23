# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real v0.30 repack, Marlin decode and Firefly CUDA prefill.

This deliberately requires the compiled Firefly extension. A Python fallback
must not turn a missing/broken CUDA extension into a passing kernel test.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires an SM75 CUDA GPU", allow_module_level=True)


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("signed_group_scales", [False, True])
@torch.inference_mode()
def test_v030_firefly_hybrid_numerics(batch_size, signed_group_scales):
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("Firefly CUDA extension targets SM75")
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        _ff_hybrid_linear,
    )
    from vllm.model_executor.layers.quantization.utils import firefly
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_empty,
        marlin_make_workspace_new,
        marlin_permute_scales,
    )

    torch.manual_seed(16)
    device = torch.device("cuda:0")
    size_k, size_n, group_size = 256, 128, 128
    # Build an independent clean INT4 matrix, packed along K exactly as GPTQ
    # stores it, then invoke the actual v0.30 native repacker (no g_idx API).
    q = torch.randint(0, 16, (size_k, size_n), device=device, dtype=torch.int64)
    shifts = torch.arange(8, device=device) * 4
    packed = (q.reshape(size_k // 8, 8, size_n) << shifts[None, :, None]).sum(1)
    marlin_weight = ops.gptq_marlin_repack(
        packed.to(torch.int32), size_k=size_k, size_n=size_n,
        num_bits=4, is_a_8bit=False,
    )
    scales = (0.01 + torch.rand(size_n, size_k // group_size, device=device) * 0.04).half()
    if signed_group_scales:
        scales[:, 1::2] *= -1
    dense_weight = (q.t().float() - 8) * scales.float().repeat_interleave(group_size, dim=1)
    marlin_scales = marlin_permute_scales(
        scales.t().contiguous(), size_k=size_k, size_n=size_n,
        group_size=group_size,
    )
    # Check native repack inversion before testing GEMMs. This catches upstream
    # layout changes independently of the approximate INT8 prefill numerics.
    torch.testing.assert_close(
        firefly.marlin_to_int4_q(marlin_weight, size_k, size_n),
        q.t().to(torch.uint8), atol=0, rtol=0,
    )
    assert firefly._load_cuda_mod() is not None, "native Firefly extension unavailable"
    quantized_weight, row_scale = firefly.dequant_marlin_to_int8(
        marlin_weight, scales, group_size, size_k, size_n,
    )
    expected_row_scale = dense_weight.abs().amax(dim=1) / 127
    torch.testing.assert_close(row_scale, expected_row_scale, atol=1e-8, rtol=1e-6)
    # Construct the ideal ratios independently in FP64 from the clean INT4
    # codes and FP16 scales. CUDA and TensorIterator may evaluate the FP32
    # division/constant reciprocal differently at exact half-integers. Permit
    # either adjacent integer only within a conservative four-FP32-epsilon error
    # envelope of such a tie; every non-boundary value must match exactly.
    dense_fp64 = (q.t().double() - 8) * scales.double().repeat_interleave(group_size, dim=1)
    ideal_scaled = dense_fp64 * 127 / dense_fp64.abs().amax(dim=1, keepdim=True)
    ideal_nearest = ideal_scaled.round().clamp(-127, 127)
    magnitude = ideal_scaled.abs()
    half_distance = (magnitude - (magnitude.floor() + 0.5)).abs()
    fp32_bound = 4 * torch.finfo(torch.float32).eps * magnitude.clamp_min(1)
    near_half = half_distance <= fp32_bound
    actual_codes = quantized_weight.double()
    torch.testing.assert_close(actual_codes[~near_half], ideal_nearest[~near_half], atol=0, rtol=0)
    assert (
        (actual_codes[near_half] == ideal_scaled[near_half].floor())
        | (actual_codes[near_half] == ideal_scaled[near_half].ceil())
    ).all()
    pytorch_codes = (dense_weight / expected_row_scale[:, None]).round().clamp(-127, 127)
    differs_from_pytorch = actual_codes != pytorch_codes
    assert near_half[differs_from_pytorch].all()
    if differs_from_pytorch.any():
        differences = actual_codes[differs_from_pytorch] - pytorch_codes[differs_from_pytorch]
        assert differences.abs().max() <= 1
    print(dict(
        diagnostic="firefly_int8_rounding", signed_group_scales=signed_group_scales,
        pytorch_mismatch=differs_from_pytorch.sum().item(),
        fp64_nearest_mismatch=(actual_codes != ideal_nearest).sum().item(),
        near_half_count=near_half.sum().item(),
        max_half_distance_of_pytorch_mismatch=(
            half_distance[differs_from_pytorch].max().item()
            if differs_from_pytorch.any() else 0.0
        ),
    ))
    empty_int = marlin_make_empty(device)
    empty_half = torch.empty(0, dtype=torch.float16, device=device)
    x = torch.randn(batch_size, size_k, device=device, dtype=torch.float16)
    actual = _ff_hybrid_linear(
        x, marlin_weight, marlin_scales, empty_int,
        marlin_make_workspace_new(device), empty_half, empty_half,
        scales, empty_int, row_scale, size_k, size_n, size_k, size_n,
        group_size, 16, False, False,
    )
    reference = torch.nn.functional.linear(x.float(), dense_weight)
    relative_rmse = (
        (actual.float() - reference).square().mean().sqrt()
        / reference.square().mean().sqrt()
    ).item()
    print(dict(
        test="firefly_v030_native", batch=batch_size,
        signed_group_scales=signed_group_scales,
        path="marlin_decode" if batch_size <= 16 else "firefly_int8_prefill",
        relative_rmse=relative_rmse,
    ))
    assert torch.isfinite(actual).all()
    # Prefill intentionally quantizes activations and weights to INT8; decode
    # uses the original INT4 values. Keep their accuracy budgets separate.
    assert relative_rmse < (0.005 if batch_size <= 16 else 0.02)