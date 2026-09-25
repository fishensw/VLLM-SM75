import torch
import triton
import triton.language as tl

@triton.jit
def _gemma_norm(X, W, Y, stride: tl.constexpr, H: tl.constexpr,
                EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride + col, col < H, 0).to(tl.float32)
    w = tl.load(W + col, col < H, 0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / H
    normalized = x * tl.rsqrt(variance + EPS)
    result = normalized * (1.0 + w)
    tl.store(Y + row * H + col, result, col < H)

def sm75_mtp_gemma_norm(x, module):
    if (not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2
        or x.stride(-1) != 1 or module.weight.dtype not in (torch.float16, torch.float32)):
        return module(x)
    h = x.shape[-1]
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    _gemma_norm[(x.shape[0],)](x, module.weight, out, x.stride(0), h,
        module.variance_epsilon, triton.next_power_of_2(h),
        num_warps=8 if h > 4096 else 4, enable_fp_fusion=False)
    return out
