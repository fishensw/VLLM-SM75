# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real per-tensor FP8 context-KV/MLP regression, without model downloads.

Compare explicit Marlin with the upstream auto choice using identical inputs
and weights. Report the actual selected kernel; do not infer it from GPU type.

Run in the built v0.1.6 GPU image:
  python -m pytest tests/model_executor/test_dflash_quantization_gpu.py -q -s

Set SM75_DFLASH_FP8_CHECKPOINT to a safetensors file (or its directory) to
also compare one complete real checkpoint MLP layer, using its actual
post-attention RMSNorm weight and epsilon for the input contract. This is a
normalized random residual, not a replay of full-model hidden states. Only
that layer is read. Raw unnormalized stress is retained separately in evidence.

This checks kernel numerics and DFlash dispatch, not full-model acceptance.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

# The desktop CPU suite deliberately has no torch dependency.
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA GPU", allow_module_level=True)

from compressed_tensors.quantization import QuantizationArgs
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A16Fp8,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model


@pytest.fixture(scope="module", autouse=True)
def singleton_model_parallel(tmp_path_factory):
    """Create real NCCL/TP groups required by production weight parameters."""
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    torch.cuda.set_device(0)
    rendezvous = tmp_path_factory.mktemp("sm75-nccl") / "rendezvous"
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=rendezvous.as_uri(),
                backend="nccl",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                backend="nccl",
            )
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


class _CompressedQKV(nn.Module):
    """Use production CT allocation, repacking and GEMM on a local QKV shard."""

    def __init__(self, input_size, q_size, kv_size, bias, partition_sizes=None):
        super().__init__()
        self.scheme = CompressedTensorsW8A16Fp8(
            QuantizationArgs(
                num_bits=8, type="float", strategy="tensor",
                symmetric=True, dynamic=False,
            ),
            is_static_input_scheme=False,
        )
        self.quant_method = self.scheme
        widths = partition_sizes or [q_size, kv_size, kv_size]
        # Match the LinearBase/MergedColumnParallelLinear metadata consumed by
        # both Humming and Marlin; the quantization kernels remain unmodified.
        self.input_size = input_size
        self.output_size = sum(widths)
        self.output_partition_sizes = widths
        self.params_dtype = torch.float16
        self.has_bias = bias is not None
        self.skip_bias_add = False
        self.prefix = "test_projection"
        self.scheme.create_weights(
            layer=self,
            input_size_per_partition=input_size,
            output_partition_sizes=widths,
            input_size=input_size,
            output_size=sum(widths),
            params_dtype=torch.float16,
            weight_loader=default_weight_loader,
        )
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x):
        return self.scheme.apply_weights(self, x, self.bias), None


@pytest.mark.parametrize("linear_backend", ["marlin", "auto"])
@pytest.mark.parametrize("q_size", [128, 256])
@pytest.mark.parametrize("batch_size", [1, 7, 32])
@pytest.mark.parametrize("has_bias", [False, True])
@torch.inference_mode()
def test_compressed_tensor_context_kv_backends(q_size, batch_size, has_bias, linear_backend):
    if torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("this regression targets GPUs using the FP8 Marlin fallback")
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    config = VllmConfig()
    config.kernel_config.linear_backend = linear_backend
    config.model_config = SimpleNamespace(dtype=torch.float16)
    try:
        with set_current_vllm_config(config):
            _check_context_kv(q_size, batch_size, has_bias, linear_backend)
    finally:
        torch.set_default_dtype(old_dtype)


def _check_context_kv(q_size, batch_size, has_bias, linear_backend):
    torch.manual_seed(13)
    k, kv_size, layers = 256, 64, 2
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = nn.Module()
    model.hidden_norm.weight = nn.Parameter(
        torch.ones(k, dtype=torch.float16, device="cuda"), requires_grad=False
    )
    model._rms_norm_eps = 1e-6
    attns, reference_weights, reference_biases = [], [], []
    for layer_index in range(layers):
        width = q_size + 2 * kv_size
        bias = (torch.randn(width, device="cuda") * 0.01) if has_bias else None
        projection = _CompressedQKV(k, q_size, kv_size, bias).cuda()
        scales = torch.tensor([0.01, 0.025, 0.04], dtype=torch.float32)
        row_scales = torch.repeat_interleave(
            scales, torch.tensor([q_size, kv_size, kv_size])
        )
        dense = torch.randn(width, k, dtype=torch.float32) * 0.05
        quantized = (dense / row_scales[:, None]).to(torch.float8_e4m3fn)
        reference_weights.append(
            (quantized.float() * row_scales[:, None])[q_size:].cuda()
        )
        reference_biases.append(bias[q_size:].float() if has_bias else None)
        projection.weight.data.copy_(quantized.cuda())
        projection.weight_scale.data.copy_(scales.cuda())
        attns.append(SimpleNamespace(
            qkv_proj=projection, q_size=q_size,
            k_norm=SimpleNamespace(weight=torch.ones(kv_size, device="cuda")),
        ))

    # Match the real load lifecycle: build buffers first, then repack in place.
    model._build_context_kv_buffers(attns, has_bias)
    assert model._fuse_context_kv is False
    assert not hasattr(model, "_fused_kv_weight")
    kernels = []
    for attn in attns:
        attn.qkv_proj.scheme.process_weights_after_loading(attn.qkv_proj)
        kernel = type(attn.qkv_proj.scheme.linear_kernel).__name__
        kernels.append(kernel)
        if linear_backend == "marlin" or torch.cuda.get_device_capability() == (7, 5):
            assert kernel == "MarlinFP8ScaledMMLinearKernel", kernel
            assert attn.qkv_proj.weight.dtype == torch.int32
        else:
            assert kernel in (
                "HummingFP8ScaledMMLinearKernel", "MarlinFP8ScaledMMLinearKernel"
            ), kernel

    x = torch.randn(batch_size, k, device="cuda", dtype=torch.float16)
    # Reuse the exact normalized input to isolate GEMM/layout errors.
    normed = model._normalize_context_states(x)
    flat_reference = torch.cat([
        torch.nn.functional.linear(normed.float(), weight, bias)
        for weight, bias in zip(reference_weights, reference_biases)
    ], dim=-1)
    reference = flat_reference.view(batch_size, layers, 2, 1, kv_size)
    reference = reference.permute(2, 1, 0, 3, 4).contiguous()
    actual = model._project_context_kv(x, batch_size, layers, 1, kv_size)
    for label, got, expected in zip(("K", "V"), actual, reference):
        assert torch.isfinite(got).all()
        relative_rmse = (
            (got.float() - expected).square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-6)
        ).item()
        print(dict(
            projection=label, q_size=q_size, batch=batch_size,
            bias=has_bias, linear_backend=linear_backend, kernels=kernels,
            relative_rmse=relative_rmse,
        ))
        # Correct FP16 Marlin accumulation is comfortably below 1%; missing
        # per-tensor scales, layout corruption or wrong exponent bias is not.
        assert relative_rmse < 0.01
        torch.testing.assert_close(got.float(), expected, atol=0.025, rtol=0.025)



@pytest.mark.parametrize("linear_backend", ["marlin", "auto"])
@pytest.mark.parametrize("batch_size", [1, 7, 32])
@torch.inference_mode()
def test_compressed_tensor_bf16_mlp_backends(batch_size, linear_backend):
    """Match issue #13: merged gate/up and down projections are FP8."""
    from vllm.model_executor.models.dflash_sm70 import DFlashSM70MLP

    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("SM75 BF16 emulation regression")
    torch.manual_seed(13)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    config = VllmConfig()
    config.kernel_config.linear_backend = linear_backend
    config.model_config = SimpleNamespace(dtype=torch.float16)
    try:
        with set_current_vllm_config(config):
            quantized, dense, kernels = [], [], []
            for input_size, widths, scales in (
                (256, [256, 256], [0.001, 0.003]),
                (256, [256], [0.002]),
            ):
                projection = _CompressedQKV(
                    input_size, 0, 0, None, partition_sizes=widths
                ).cuda()
                s = torch.tensor(scales, dtype=torch.float32)
                rows = torch.repeat_interleave(s, torch.tensor(widths))
                weight = torch.randn(sum(widths), input_size, dtype=torch.float32)
                weight = (weight * 0.01 / rows[:, None]).to(torch.float8_e4m3fn)
                reference = (weight.float() * rows[:, None]).half().cuda()
                projection.weight.data.copy_(weight.cuda())
                projection.weight_scale.data.copy_(s.cuda())
                projection.scheme.process_weights_after_loading(projection)
                kernel = type(projection.scheme.linear_kernel).__name__
                kernels.append(kernel)
                if linear_backend == "marlin" or torch.cuda.get_device_capability() == (7, 5):
                    assert kernel == "MarlinFP8ScaledMMLinearKernel", kernel
                else:
                    assert kernel in (
                        "HummingFP8ScaledMMLinearKernel", "MarlinFP8ScaledMMLinearKernel"
                    ), kernel
                quantized.append(projection)
                dense.append(reference)

            class DenseProjection(nn.Module):
                def __init__(self, weight):
                    super().__init__()
                    self.weight = nn.Parameter(weight, requires_grad=False)

                def forward(self, x):
                    return torch.nn.functional.linear(x, self.weight), None

            actual_mlp = DFlashSM70MLP(SimpleNamespace(
                gate_up_proj=quantized[0], down_proj=quantized[1]
            ))
            reference_mlp = DFlashSM70MLP(SimpleNamespace(
                gate_up_proj=DenseProjection(dense[0]),
                down_proj=DenseProjection(dense[1]),
            ))
            x = torch.randn(batch_size, 256, device="cuda", dtype=torch.float16)
            expected = reference_mlp(x).float()
            actual = actual_mlp(x).float()
            print(dict(
                diagnostic="mlp_finiteness", linear_backend=linear_backend,
                actual_nonfinite=(~torch.isfinite(actual)).sum().item(),
                expected_nonfinite=(~torch.isfinite(expected)).sum().item(),
                actual_max_finite=actual[torch.isfinite(actual)].abs().max().item(),
                expected_max_finite=expected[torch.isfinite(expected)].abs().max().item(),
            ))
            assert torch.isfinite(expected).all(), "dense reference overflow"
            assert torch.isfinite(actual).all(), "quantized backend overflow"
            relative_rmse = (
                (actual - expected).square().mean().sqrt()
                / expected.square().mean().sqrt().clamp_min(1e-8)
            ).item()
            print(dict(test="bf16_mlp", batch=batch_size,
                       linear_backend=linear_backend, kernels=kernels,
                       relative_rmse=relative_rmse))
            # Small differences in accumulation can cross BF16 rounding points.
            assert relative_rmse < 0.03
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize("linear_backend", ["marlin", "auto"])
@pytest.mark.parametrize("batch_size", [1, 7])
@torch.inference_mode()
def test_real_checkpoint_bf16_mlp_backends(batch_size, linear_backend):
    """Compare layer-0 weights after its real post-attention RMSNorm contract."""
    from safetensors import safe_open
    from vllm.model_executor.models.dflash_sm70 import DFlashSM70MLP

    checkpoint_env = os.environ.get("SM75_DFLASH_FP8_CHECKPOINT")
    if not checkpoint_env:
        pytest.skip("set SM75_DFLASH_FP8_CHECKPOINT for real-checkpoint diagnostics")
    checkpoint = Path(checkpoint_env)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.safetensors"
    assert checkpoint.is_file(), checkpoint
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("SM75 BF16 emulation regression")

    torch.manual_seed(13)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    config = VllmConfig()
    config.kernel_config.linear_backend = linear_backend
    config.model_config = SimpleNamespace(dtype=torch.float16)
    try:
        with set_current_vllm_config(config), safe_open(
            checkpoint, framework="pt", device="cpu"
        ) as tensors:
            prefix = "layers.0.mlp."
            gate_shape = tensors.get_slice(prefix + "gate_proj.weight").get_shape()
            up_shape = tensors.get_slice(prefix + "up_proj.weight").get_shape()
            down_shape = tensors.get_slice(prefix + "down_proj.weight").get_shape()
            assert gate_shape == up_shape
            intermediate_size, hidden_size = gate_shape
            assert down_shape == [hidden_size, intermediate_size]
            quantized, dense, kernels, actual_scales = [], [], [], {}
            for names, input_size, widths in (
                (("gate_proj", "up_proj"), hidden_size,
                 [intermediate_size, intermediate_size]),
                (("down_proj",), intermediate_size, [hidden_size]),
            ):
                projection = _CompressedQKV(
                    input_size, 0, 0, None, partition_sizes=widths
                ).cuda()
                reference_parts = []
                output_offset = 0
                scales = []
                for name, width in zip(names, widths):
                    # safe_open is memory mapped; only this projection is read.
                    weight = tensors.get_slice(prefix + name + ".weight")[:]
                    scale = tensors.get_tensor(prefix + name + ".weight_scale")
                    assert weight.dtype == torch.float8_e4m3fn
                    assert scale.numel() == 1
                    scale_value = scale.float().item()
                    scales.append(scale_value)
                    actual_scales[name] = scale_value
                    projection.weight.data[output_offset:output_offset + width].copy_(
                        weight.cuda()
                    )
                    reference_parts.append(
                        (weight.float() * scale_value).half().cuda()
                    )
                    output_offset += width
                projection.weight_scale.data.copy_(
                    torch.tensor(scales, device="cuda", dtype=torch.float32)
                )
                projection.scheme.process_weights_after_loading(projection)
                kernel = type(projection.scheme.linear_kernel).__name__
                if linear_backend == "marlin" or torch.cuda.get_device_capability() == (7, 5):
                    assert kernel == "MarlinFP8ScaledMMLinearKernel", kernel
                else:
                    assert kernel in (
                        "HummingFP8ScaledMMLinearKernel",
                        "MarlinFP8ScaledMMLinearKernel",
                    ), kernel
                kernels.append(kernel)
                quantized.append(projection)
                dense.append(torch.cat(reference_parts, dim=0))

            class DenseProjection(nn.Module):
                def __init__(self, weight):
                    super().__init__()
                    self.weight = nn.Parameter(weight, requires_grad=False)

                def forward(self, x):
                    return torch.nn.functional.linear(x, self.weight), None

            actual_mlp = DFlashSM70MLP(SimpleNamespace(
                gate_up_proj=quantized[0], down_proj=quantized[1]
            ))
            reference_mlp = DFlashSM70MLP(SimpleNamespace(
                gate_up_proj=DenseProjection(dense[0]),
                down_proj=DenseProjection(dense[1]),
            ))
            # The MLP receives RMSNorm output, not a unit-variance random
            # vector. Use the checkpoint gamma and model epsilon, with the
            # same BF16 rounding as DFlashSM70RMSNorm._reference.
            model_config = json.loads(
                checkpoint.with_name("config.json").read_text(encoding="utf-8")
            )
            epsilon = float(model_config["rms_norm_eps"])
            gamma = tensors.get_tensor(
                "layers.0.post_attention_layernorm.weight"
            ).cuda().float().to(torch.bfloat16).float()
            assert gamma.numel() == hidden_size
            residual = torch.randn(
                batch_size, hidden_size, device="cuda", dtype=torch.float16
            ).float()
            normalized = residual * torch.rsqrt(
                residual.square().mean(dim=-1, keepdim=True) + epsilon
            )
            x = (normalized * gamma).to(torch.bfloat16).to(torch.float16)
            expected = reference_mlp(x).float()
            actual = actual_mlp(x).float()
            print(dict(
                diagnostic="mlp_finiteness", linear_backend=linear_backend,
                actual_nonfinite=(~torch.isfinite(actual)).sum().item(),
                expected_nonfinite=(~torch.isfinite(expected)).sum().item(),
                actual_max_finite=actual[torch.isfinite(actual)].abs().max().item(),
                expected_max_finite=expected[torch.isfinite(expected)].abs().max().item(),
            ))
            assert torch.isfinite(expected).all(), "dense reference overflow"
            assert torch.isfinite(actual).all(), "quantized backend overflow"
            relative_rmse = (
                (actual - expected).square().mean().sqrt()
                / expected.square().mean().sqrt().clamp_min(1e-8)
            ).item()
            print(dict(
                test="real_checkpoint_bf16_mlp", layer=0,
                checkpoint=checkpoint.name, batch=batch_size,
                linear_backend=linear_backend, kernels=kernels,
                scales=actual_scales, relative_rmse=relative_rmse,
                input_contract="checkpoint_post_attention_rmsnorm",
                gamma_rms=gamma.square().mean().sqrt().item(),
                input_rms=x.float().square().mean().sqrt().item(),
            ))
            assert relative_rmse < 0.03
    finally:
        torch.set_default_dtype(old_dtype)
