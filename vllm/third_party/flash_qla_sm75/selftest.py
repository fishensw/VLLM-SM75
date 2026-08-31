# SPDX-License-Identifier: Apache-2.0
"""Load and minimally execute the image-owned SM75 FlashQLA extension."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)

from .fused_fwd import chunk_gated_delta_rule_fwd_sm70_vlk_varlen


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SM75 FlashQLA self-test requires CUDA.")
    capability = torch.cuda.get_device_capability(0)
    if capability != (7, 5):
        raise RuntimeError(f"Expected SM75, got compute capability {capability}.")

    device = torch.device("cuda:0")
    tokens = 64
    q_heads = 8
    v_heads = 16
    dim = 128
    torch.manual_seed(1234)
    q = (
        F.normalize(torch.randn(1, tokens, q_heads, dim, device=device).float(), dim=-1)
        .to(torch.float16)
        .contiguous()
    )
    k = (
        F.normalize(torch.randn(1, tokens, q_heads, dim, device=device).float(), dim=-1)
        .to(torch.float16)
        .contiguous()
    )
    v = torch.randn(
        1, tokens, v_heads, dim, device=device, dtype=torch.float16
    ).contiguous()
    g = (-torch.rand(1, tokens, v_heads, device=device) * 0.05).contiguous()
    beta = torch.rand(1, tokens, v_heads, device=device).contiguous()
    state = torch.zeros(1, v_heads, dim, dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    with torch.inference_mode():
        reference_output, reference_state = fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            initial_state=state.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        output, final_state = chunk_gated_delta_rule_fwd_sm70_vlk_varlen(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            initial_state=state.clone(),
            output_final_state=True,
            validate_cu_seqlens=False,
        )
    assert final_state is not None
    assert reference_state is not None
    if not torch.isfinite(output).all() or not torch.isfinite(final_state).all():
        raise RuntimeError("SM75 FlashQLA self-test produced non-finite values.")
    output_error = (output.float() - reference_output.float()).abs()
    state_error = (final_state.float() - reference_state.float()).abs()
    torch.testing.assert_close(
        output.float(), reference_output.float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        final_state.float(), reference_state.float(), atol=3e-2, rtol=3e-2
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "device": torch.cuda.get_device_name(0),
                "capability": list(capability),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "output_shape": list(output.shape),
                "state_shape": list(final_state.shape),
                "reference": "vllm_triton_fla",
                "tolerance": {"atol": 0.03, "rtol": 0.03},
                "output_max_abs_error": output_error.max().item(),
                "output_mean_abs_error": output_error.mean().item(),
                "state_max_abs_error": state_error.max().item(),
                "state_mean_abs_error": state_error.mean().item(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
