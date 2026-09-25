"""Experimental HC-only SM75 dispatch. Disabled unless explicitly requested.

Install only on the three HC projection modules, never globally on Linear.
An opaque op keeps prefill shape branches outside torch.compile tracing.
"""
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_seen_shapes = set()


@triton.jit
def _row_gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
              R: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, BK)
    x = tl.load(X + cols, cols < K, 0).to(tl.float32)
    w = tl.load(W + rows[:, None] * K + cols[None, :],
                (rows[:, None] < N) & (cols[None, :] < K), 0).to(tl.float32)
    values = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + rows, values, rows < N)


@torch.library.custom_op('next_hc::gemv', mutates_args=())
def hc_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    eligible = (x.is_cuda and weight.is_cuda and x.device == weight.device
                and x.dtype == weight.dtype == torch.float16
                and x.is_contiguous() and weight.is_contiguous()
                and x.shape[-1] == k and x.numel() == k
                and (n, k) in ((336, 10240), (320, 10240), (10240, 320))
                and torch.cuda.get_device_capability(x.device) == (7, 5))
    if not eligible:
        return F.linear(x, weight)
    if (n, k) not in _seen_shapes:
        _seen_shapes.add((n, k))
        print(f'NEXT_HC_GEMV_ACTIVE pid={os.getpid()} N={n} K={k} M=1', flush=True)
    output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    rows, warps = (4, 4) if k == 320 else (1, 8)
    _row_gemv[(triton.cdiv(n, rows),)](
        x, weight, output, n, k, rows, triton.next_power_of_2(k), num_warps=warps)
    return output


@hc_gemv.register_fake
def _fake(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _dispatch(layer, x, weight, bias=None):
    if bias is not None:
        return F.linear(x, weight, bias)
    return hc_gemv(x, weight)


def install_hc_candidate(hc):
    if os.environ.get('VLLM_SM75_QWEN38_HC_GEMV', '0') != '1':
        return
    from vllm import envs
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    if envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError('HC candidate would be bypassed by batch-invariant dispatch')
    names = ('input_mix_weight_down_block_inject', 'input_mix_weight_down', 'input_mix_weight_up')
    for name in names:
        layer = getattr(hc, name, None)
        if layer is None:
            continue
        if type(layer.quant_method) is not UnquantizedLinearMethod:
            raise RuntimeError(f'Unexpected HC linear method: {name}')
        layer.quant_method._gemm_impl = _dispatch
