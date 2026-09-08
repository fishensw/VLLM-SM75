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
import vllm.envs as envs

logger = logging.getLogger(__name__)


def firefly_active_int4() -> bool:
    """int4(W4A16) 是否走 firefly prefill: VLLM_FIREFLY 开(1/auto 等价)。"""
    return envs.VLLM_FIREFLY == "1"


def firefly_active_fp8() -> bool:
    """fp8(W8A8) prefill 是否走 firefly: VLLM_FIREFLY 开 且 VLLM_FIREFLY_FUSED 显式
    选了 fused(1)或非 fused(0)(非 auto)。

    auto(默认)= 选最快 = 当前关闭(sm75 实测 firefly-fp8 不比 marlin 快, 走 marlin;
    fp8 加速走 VLLM_FIREFLY_AR, 见 PLAN-fp8-allreduce)。1 = 强制 fused; 0 = 强制
    非 fused。int4 不受 VLLM_FIREFLY_FUSED 影响(只有非 fused)。
    """
    return envs.VLLM_FIREFLY == "1" and envs.VLLM_FIREFLY_FUSED in ("1", "0")

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


# ---- fp8 fuse: GEMM B-load 内即时反量化 fp8->int8(省 int8 缓存, 1B/参数) ----
# 与 int8_prefill_linear 逐 bit 一致(同 mma.s8 路径 + 同 def 反量化), 见 firefly_fused.cu。
# 懒加载 firefly_fused(torch.utils.cpp_extension.load, 首次 fused prefill 时 JIT,
# 需镜像内 flashinfer cutlass 头 + 本目录 _cutlass_ext 头; 失败回退 int8 路径)。
_fused_mod = None
_fused_load_attempted = False


def _find_fused_include_dirs() -> list[str] | None:
    """定位 firefly_fused.cu 编译所需 include, 找不到返回 None。

    需要两处:
      1. flashinfer cutlass(2.x API + cute + fp8), 探针常见安装位置;
      2. 本目录 _cutlass_ext(打包的 vllm cutlass_extensions/epilogue 头)。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(here, "_cutlass_ext")
    ext_h = os.path.join(
        ext_dir, "cutlass_extensions", "epilogue", "broadcast_load_epilogue_c2x.hpp"
    )
    if not os.path.isfile(ext_h):
        return None
    candidates = [
        "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include",
    ]
    try:
        import flashinfer

        fi = os.path.join(
            os.path.dirname(os.path.abspath(flashinfer.__file__)),
            "data", "cutlass", "include",
        )
        if fi not in candidates:
            candidates.append(fi)
    except Exception:  # noqa: BLE001
        pass
    cutlass_inc = next((c for c in candidates if os.path.isdir(c)), None)
    if cutlass_inc is None:
        return None
    return [cutlass_inc, ext_dir]


def _load_fused_mod():
    """懒加载 firefly_fused: 优先构建期预编译 .so(免运行时 JIT), 回退 JIT, 失败 None。"""
    global _fused_mod, _fused_load_attempted
    if _fused_load_attempted:
        return _fused_mod
    _fused_load_attempted = True
    # 1. 构建期预编译 .so(镜像内置, 免首测 ~5min JIT; 见 firefly-fused-builder stage)
    prebuilt = os.environ.get("VLLM_FIREFLY_FUSED_PREBUILT_PATH")
    if prebuilt and os.path.isfile(prebuilt):
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("firefly_fused", prebuilt)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _fused_mod = module
                logger.info(
                    "firefly fused kernel loaded (prebuilt %s)", prebuilt
                )
                return _fused_mod
        except Exception as e:  # noqa: BLE001 - 预编译加载失败回退 JIT
            logger.warning("firefly fused prebuilt load failed, try JIT: %s", e)
    # 2. JIT 回退(单 arch sm_75; 开发/cp-in 场景)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cu_path = os.path.join(here, "firefly_fused.cu")
        if not os.path.exists(cu_path):
            logger.warning("firefly_fused source not found at %s; skip fused", cu_path)
            return None
        inc = _find_fused_include_dirs()
        if inc is None:
            logger.warning("firefly_fused: cutlass include not found; skip fused")
            return None
        from torch.utils.cpp_extension import load as _load_ext

        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
        _fused_mod = _load_ext(
            name="firefly_fused",
            sources=[cu_path],
            extra_cuda_cflags=["-O3"],
            extra_include_paths=inc,
            verbose=False,
        )
        logger.info("firefly fused kernel loaded (fp8 B-load dequant)")
    except Exception as e:  # noqa: BLE001 - 编译/无 CUDA 回退 int8
        logger.warning("firefly fused ext load failed, fallback to int8: %s", e)
        _fused_mod = None
    return _fused_mod


def compute_c_n_fp8(w_fp8: torch.Tensor, s_inv: torch.Tensor) -> torch.Tensor:
    """fp8(e4m3)+128x128 block scale -> per-channel c_n [N] fp32。

    只做 fp8_to_int8 的 c_n pass(amax_k|w_deq|/127), 不物化 w_int8 —— fuse
    路径省 int8 缓存(1B/参数), w_deq 在 GEMM B-load 里现算。
    """
    N, K = w_fp8.shape
    w_deq = w_fp8.to(torch.float32)
    s_exp = (
        s_inv.to(torch.float32)
        .repeat_interleave(128, 0)
        .repeat_interleave(128, 1)[:N, :K]
    )
    w_deq = w_deq * s_exp
    return w_deq.abs().amax(dim=1) / 127.0


def fp8_fused_prefill_linear(
    x: torch.Tensor,
    w_fp8: torch.Tensor,  # [N, K] fp8 e4m3 行主序
    s_inv: torch.Tensor,  # [N/128, K/128] block scale
    c_n: torch.Tensor,  # [N]
) -> torch.Tensor:
    """x [*, K] -> per-token 动态 int8 -> fused GEMM(B-load 内 fp8->int8) -> [*, N]。

    与 int8_prefill_linear 逐 bit 一致(同 mma 路径 + 同 def 反量化)。
    """
    orig = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    x_q, x_s, _ = ops.scaled_int8_quant(x2d.contiguous())
    mod = _load_fused_mod()
    y = mod.fp8_fused_scaled_mm(x_q, w_fp8, s_inv, c_n, x_s, x.dtype)
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


# ---- torch.compile(fullgraph) 兼容 ----
# 裸 pybind 反量化 kernel dynamo 追不了: VLLM_COMPILE(fullgraph=True, 即去掉
# --enforce-eager 后) 追到 prefill 分支会抛 "Unsupported: Attempted to call
# function marked as skipped"(gb0007)。注册成 torch.library op 后 dynamo 当
# 不透明 op 处理, 兼容 fullgraph 编译 + CUDA graph capture。CUDA mod 仍懒加载:
# op body 首次执行才 _load_cuda_mod()(JIT 编译), 不拖 vllm import。
@torch.library.custom_op("firefly::dequant_marlin_cached", mutates_args=())
def _ff_dequant_marlin_cached_op(
    marlin_w: torch.Tensor,
    weight_scale: torch.Tensor,
    zp_t: torch.Tensor,
    c_n: torch.Tensor,
    n: int,
    k: int,
    pn: int,
    pk: int,
    gs: int,
    use_recip: bool,
) -> torch.Tensor:
    mod = _load_cuda_mod()
    out_int8 = torch.empty((n, k), dtype=torch.int8, device=marlin_w.device)
    kernel = (
        mod.firefly_dequant_cached_recip if use_recip else mod.firefly_dequant_cached
    )
    kernel(marlin_w, weight_scale, zp_t, out_int8, c_n, n, k, pn, pk, gs)
    return out_int8


@_ff_dequant_marlin_cached_op.register_fake
def _(_marlin_w, _weight_scale, _zp_t, _c_n, n, k, _pn, _pk, _gs, _use_recip):
    return torch.empty((n, k), dtype=torch.int8, device=_marlin_w.device)


def dequant_marlin_to_int8_cached(
    marlin_w: torch.Tensor,
    weight_scale: torch.Tensor,  # 干净 scale [N, K/group_size] (N 在前)
    c_n: torch.Tensor,  # [N] 预计算 per-channel scale(load 时算一次)
    group_size: int,
    size_k: int,
    size_n: int,
    padded_k: int | None = None,
    padded_n: int | None = None,
    w_zp: torch.Tensor | None = None,  # None=对称(zp=8); AWQ qzeros [N/8, K/gs]
    use_recip: bool = False,
) -> torch.Tensor:
    """B1-hard 单遍反量化: c_n 已预计算, 只跑量化 pass(免两遍 amax)。

    use_recip=False: 除法, 与 dequant_marlin_to_int8 的 pass2 逐 bit 一致。
    use_recip=True : 乘倒数(fast), 更快但 off-by-one ≤0.06%。
    优先 CUDA kernel(firefly_dequant_cached[_recip]); 编译失败/无 GPU 回退 PyTorch 版。
    返回 w_int8 [N, K] int8。
    """
    N, K = size_n, size_k
    if _load_cuda_mod() is not None and marlin_w.is_cuda:
        pk = padded_k if padded_k is not None else K
        pn = padded_n if padded_n is not None else N
        zp_t = (
            torch.empty(0, dtype=torch.float32, device=marlin_w.device)
            if w_zp is None
            else _unpack_awq_zp(w_zp, N, weight_scale.shape[1])
        )
        # 走 torch.library op(见上定义), 兼容 VLLM_COMPILE fullgraph。
        return _ff_dequant_marlin_cached_op(
            marlin_w, weight_scale, zp_t, c_n, N, K, pn, pk, group_size, use_recip
        )

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
    if use_recip:
        r = torch.where(c_n > 0, 1.0 / c_n, 0.0)
        return torch.clamp(torch.round(w_deq * r.unsqueeze(1)), -127, 127).to(torch.int8)
    return torch.clamp(
        torch.round(w_deq / c_n.unsqueeze(1)), -127, 127
    ).to(torch.int8)
