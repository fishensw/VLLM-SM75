# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""firefly(int4→int8 prefill)混合方案辅助函数(SM75)。

思路: decode 保持 int4 Marlin(fp16 激活, 正确处理负 group scale);
prefill 大 M 时把 int4 权重反量化成 int8(per-group -> per-channel 折叠),
配 per-token 动态 int8 激活走 cutlass_scaled_mm(SM75 上即 IMMA tensor core)。

数学(与 P0 bench 验证一致, tmp/sm75-int8-verify/docker/p0_dequant_bench.py):
    w_deq[n,k] = (q[n,k] - 8) * s_g[n,k//gs]   # s_g 可负(AutoRound ~50% 负)
    c_n        = amax_k |w_deq[n,k]| / 127      # per-channel 正 scale
    w_int8[n,k]= clamp(round(w_deq[n,k] / c_n), -127, 127)
负 s_g 由 int8 码承载, c_n 取绝对值, 符号正确。

int8_prefill_linear(激活量化 + GEMM)由 int4/fp8 两路共用;
fp8 权重另走 fp8_to_int8 反量化(fp8 对称无 zp, 比 int4 简单)。
"""

import logging
import os

import torch

from vllm import _custom_ops as ops

logger = logging.getLogger(__name__)

# B1-hard 的 reverse-repack+反量化 CUDA kernel(独立 torch extension, 首次用时编译)。
# 编译失败(无 nvcc/CUDA)或无 GPU 时回退 PyTorch 版(正确但慢 ~50ms/层)。
_cuda_mod = None
_cuda_load_attempted = False


def _load_cuda_mod():
    """懒加载 firefly.cu(torch.utils.cpp_extension.load), 失败返回 None。

    只在首次调用时编译(缓存到 torch extensions 目录), 不拖慢 vllm import。
    """
    global _cuda_mod, _cuda_load_attempted
    if _cuda_load_attempted:
        return _cuda_mod
    _cuda_load_attempted = True
    try:
        cu_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "firefly.cu"
        )
        if not os.path.exists(cu_path):
            logger.warning(
                "firefly kernel source (firefly.cu) not found at %s; "
                "use PyTorch path",
                cu_path,
            )
            return None
        from torch.utils.cpp_extension import load as _load_ext

        # 只编 sm_75(T10)。本 overlay 面向 SM75, 强制单 arch, 首次 prefill
        # JIT 编译 ~2s(而非基础镜像多 arch 默认的 ~50s)。
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        _cuda_mod = _load_ext(
            name="firefly_cuda",
            sources=[cu_path],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        logger.info("firefly CUDA kernel loaded (B1-hard fast path)")
    except Exception as e:  # noqa: BLE001 - 编译/无 CUDA 环境回退 PyTorch
        logger.warning(
            "firefly CUDA ext load failed, fallback to PyTorch: %s", e
        )
        _cuda_mod = None
    return _cuda_mod


def _unpack_awq_zp(
    qzeros_packed: torch.Tensor, size_n: int, scale_cols: int
) -> torch.Tensor:
    """AWQ 干净 qzeros [N/8, K/gs] int32(nibble j = N offset j) -> 干净 zp [N, K/gs] float。

    镜像 auto_awq._convert_awq_to_standard_format 的 qzeros 打包:
    new_qz[n//8, g] = Σ_j zp[n//8*8+j, g] << 4j, 故 nibble j = N offset j。
    """
    shifts = torch.arange(0, 32, 4, device=qzeros_packed.device)
    zp = (qzeros_packed.long().unsqueeze(-1) >> shifts) & 0xF  # [N/8, K/gs, 8]
    zp = zp.permute(0, 2, 1).reshape(size_n, scale_cols)  # [N, K/gs]
    return zp.float()


def dequant_clean_int4_to_int8(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int = 128,
    w_zp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """干净布局 int4 -> int8 权重 + per-channel scale。

    weight_packed: [N, K/8] int32(N 在前; AWQ 已在 marlin.py 快照时转置归一化)
    weight_scale:  [N, K/group_size] fp16/bf16(可负; N 在前)
    w_zp:          None=对称(zp=8, uint4b8); AWQ 干净 qzeros [N/8, K/gs] packed

    反量化 w_deq=(q-zp)*s; 对称 zp=8, 非对称 zp=per-group qzeros 值。
    返回 (w_int8 [N, K] int8, c_n [N] fp32)。
    """
    N, Kd8 = weight_packed.shape
    K = Kd8 * 8
    wp = weight_packed.long()
    nibbles = [(wp >> (i * 4)) & 0xF for i in range(8)]  # 每个 [N, K/8]
    q = torch.stack(nibbles, dim=2).reshape(N, K).float()  # [N, K]
    s = weight_scale.float().repeat_interleave(group_size, dim=1)  # [N, K]
    zp = (
        8.0
        if w_zp is None
        else _unpack_awq_zp(w_zp, N, weight_scale.shape[1]).repeat_interleave(
            group_size, dim=1
        )
    )
    w_deq = (q - zp) * s  # [N, K]
    c_n = w_deq.abs().amax(dim=1) / 127.0  # [N] 正
    w_int8 = torch.clamp(
        torch.round(w_deq / c_n.unsqueeze(1)), -127, 127
    ).to(torch.int8)
    return w_int8, c_n


def fp8_to_int8(
    w_fp8: torch.Tensor,  # [N, K] fp8 e4m3 clean
    s_inv: torch.Tensor,  # [N/128, K/128] 128x128 block scale
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8(e4m3)+128x128 block scale -> per-channel int8 权重 + 正 scale。

    fp8 对称无 zp, 比 AWQ 非对称 int4 简单(无 (q-zp) 项):
        w_deq[n,k] = fp8_decode(w_fp8[n,k]) * s_inv[n//128, k//128]
        c_n[n]     = amax_k |w_deq[n,k]| / 127
        w_int8[n,k]= clamp(round(w_deq[n,k] / c_n[n]), -127, 127)
    PyTorch 版(小模型 load 时一次, 开销可接受); 27B 量级需 CUDA 版。
    返回 (w_int8 [N, K] int8 行主序, c_n [N] fp32)。
    """
    N, K = w_fp8.shape
    w_deq = w_fp8.to(torch.float32)  # e4m3 -> fp32 正确解码(符号/指数)
    s_exp = (
        s_inv.to(torch.float32)
        .repeat_interleave(128, 0)
        .repeat_interleave(128, 1)[:N, :K]
    )
    w_deq = w_deq * s_exp  # [N, K] 真实权重
    c_n = w_deq.abs().amax(dim=1) / 127.0  # [N] 正
    w_int8 = torch.clamp(
        torch.round(w_deq / c_n.unsqueeze(1)), -127, 127
    ).to(torch.int8)
    return w_int8, c_n


def int8_prefill_linear(
    x: torch.Tensor,
    w_int8: torch.Tensor,  # [N, K]
    c_n: torch.Tensor,  # [N]
) -> torch.Tensor:
    """x [*, K] -> per-token 动态 int8 -> cutlass_scaled_mm -> [*, N](x.dtype)。

    cutlass 要求 b 列主序(w_int8.t(), stride(0)==1, 勿 contiguous)。
    """
    orig = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    x_q, x_s, _ = ops.scaled_int8_quant(x2d.contiguous())
    y = ops.cutlass_scaled_mm(
        x_q, w_int8.t(), scale_a=x_s, scale_b=c_n, out_dtype=x.dtype
    )
    return y.reshape(*orig[:-1], -1)


# ---- B1-hard: 从 marlin 布局现反回干净 int4(不存干净 int4 副本) ----
# 镜像 csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu 的
# repack_tile, 仅 int4/fp16 场景(num_bits=4, has_perm=false, is_a_8bit=false):
#   输出 tile 位置 p, 字节内 nibble i 对应干净 (dk, dn):
#     a=p//4, b=p%4; tc_row=(a%4)*2, cur_n=b*16+a//4
#     dk=tc_row+8*(i%2)+(i//4);  dn=cur_n+8*((i//2)%2)
#   反解 (dk, dn) -> (i, p):
#     dk_lo=dk%8, dk_hi=dk//8, tc_row=dk_lo&~1, dk_off_lo=dk_lo&1
#     dn_off8=(dn%16>=8); cur_n=dn-8*dn_off8
#     i=4*dk_off_lo+2*dn_off8+dk_hi;  p=((cur_n%16)*4+tc_row//2)*4+(cur_n//16)
#   tile 顺序: tile_idx=k_tile*(N//64)+n_tile, 每 tile 128 个 int32。
# round-trip 已在 tmp/sm75-int8-verify/docker/p2_reverse_repack_test.py 逐 bit 验证。

def _reverse_repack_indices(size_k: int, size_n: int, device) -> tuple[
    torch.Tensor, torch.Tensor
]:
    """对每个干净 (n, k) 预计算它在 marlin 扁平张量里的位置(src_flat)和位移(shift)。"""
    K, N = size_k, size_n
    n_tiles = N // 64
    n = torch.arange(N, device=device)[:, None]  # [N, 1]
    k = torch.arange(K, device=device)[None, :]  # [1, K]
    nt = n // 64
    dn = n % 64
    kt = k // 16
    dk = k % 16
    dk_lo = dk % 8
    dk_hi = dk // 8
    tc_row = dk_lo & ~1  # 向下取偶
    dk_off_lo = dk_lo & 1
    dn_off8 = (dn % 16 >= 8).long()
    cur_n = dn - dn_off8 * 8
    i = 4 * dk_off_lo + 2 * dn_off8 + dk_hi
    th_id = (cur_n % 16) * 4 + tc_row // 2
    p = th_id * 4 + (cur_n // 16)
    tile_idx = kt * n_tiles + nt
    src_flat = (tile_idx * 128 + p).long()  # [N, K]
    shift = (4 * i).long()  # [N, K]
    return src_flat, shift


def marlin_to_int4_q(
    marlin_w: torch.Tensor,
    size_k: int,
    size_n: int,
    padded_k: int | None = None,
    padded_n: int | None = None,
) -> torch.Tensor:
    """marlin int4 [padded_k/16, padded_n*2] int32 -> 干净 int4 值 q [size_n, size_k]。

    反 gptq_marlin_repack(int4/fp16, 非 act-order)。padded_* 缺省取 size_*(无 padding)。
    有 tile padding 时先用 padded 维度反回再裁到 size。
    """
    pk = padded_k if padded_k is not None else size_k
    pn = padded_n if padded_n is not None else size_n
    M_flat = marlin_w.reshape(-1).long()
    src_flat, shift = _reverse_repack_indices(pk, pn, marlin_w.device)
    q_pad = ((M_flat[src_flat] >> shift) & 0xF).to(torch.uint8)  # [padded_n, padded_k]
    return q_pad[:size_n, :size_k]


def dequant_marlin_to_int8(
    marlin_w: torch.Tensor,
    weight_scale: torch.Tensor,  # 干净 scale [N, K/group_size] (N 在前)
    group_size: int,
    size_k: int,
    size_n: int,
    padded_k: int | None = None,
    padded_n: int | None = None,
    w_zp: torch.Tensor | None = None,  # None=对称(zp=8); AWQ qzeros [N/8, K/gs]
) -> tuple[torch.Tensor, torch.Tensor]:
    """B1-hard: marlin int4 + 干净 scale (+zp) -> (w_int8 [N, K], c_n [N])。

    反量化 w_deq=(q-zp)*s; 对称 zp=8(uint4b8), 非对称(AWQ) zp=per-group qzeros。
    优先 CUDA kernel(firefly_dequant, 寄存器现算 marlin 位置, ~ms 级);
    编译失败/无 GPU 时回退 PyTorch 版(正确但 ~50ms/层)。不缓存 int8。
    """
    N, K = size_n, size_k
    mod = _load_cuda_mod()
    if mod is not None and marlin_w.is_cuda:
        pk = padded_k if padded_k is not None else K
        pn = padded_n if padded_n is not None else N
        # zp: 对称→空张量(kernel 取 8.0f); 非对称→干净 [N, K/gs] float
        zp_t = (
            torch.empty(0, dtype=torch.float32, device=marlin_w.device)
            if w_zp is None
            else _unpack_awq_zp(w_zp, N, weight_scale.shape[1])
        )
        out_int8 = torch.empty((N, K), dtype=torch.int8, device=marlin_w.device)
        c_n = torch.empty((N,), dtype=torch.float32, device=marlin_w.device)
        mod.firefly_dequant(
            marlin_w,
            weight_scale,
            zp_t,
            out_int8,
            c_n,
            N,
            K,
            pn,
            pk,
            group_size,
        )
        return out_int8, c_n

    # PyTorch 回退路径
    q = marlin_to_int4_q(marlin_w, size_k, size_n, padded_k, padded_n)  # [N, K]
    s = weight_scale.float().repeat_interleave(group_size, dim=1)  # [N, K]
    zp = (
        8.0
        if w_zp is None
        else _unpack_awq_zp(w_zp, N, weight_scale.shape[1]).repeat_interleave(
            group_size, dim=1
        )
    )
    w_deq = (q.float() - zp) * s  # [N, K]
    c_n = w_deq.abs().amax(dim=1) / 127.0  # [N] 正
    w_int8 = torch.clamp(
        torch.round(w_deq / c_n.unsqueeze(1)), -127, 127
    ).to(torch.int8)
    return w_int8, c_n
