# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FireflyAllReduce: fp8 2-GPU allreduce (SHM backend, PHB/无 P2P 如 T10)。

TP2 每层 2 次 AllReduce([M,H] fp16 residual stream) 在 PCIe 瓶颈上量减半
(fp16 -> fp8), 缓解 27B prefill 通信主导 (PLAN-fp8-allreduce.md)。

传输: /dev/shm + cudaHostRegister (两进程 map 同一 tmpfs 区, 各自 register 成
pinned 设备可访问, 物理页经 tmpfs 共享)。POC 验证 vfio 下可行。

只支持 world_size=2 (TP2)、fp16 输入、SHM backend (无 P2P, 如 T10)。
P2P backend (有 P2P 机器) 后加。

env:
  VLLM_FIREFLY_AR (auto/0/fp8, auto 跟随 VLLM_FIREFLY)
  VLLM_FIREFLY_AR_MAX_SIZE (只对小/中消息走本 AR, 大消息回退 NCCL)
"""

import ctypes
import logging
import os
import time

import torch

import vllm.envs as envs

logger = logging.getLogger(__name__)

_LIB = ctypes.CDLL("libcudart.so")
_LIB.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_LIB.cudaHostRegister.restype = ctypes.c_int
_LIB.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_LIB.cudaHostUnregister.restype = ctypes.c_int


def firefly_ar_active() -> bool:
    """fp8 allreduce 是否启用: VLLM_FIREFLY_AR (auto 跟随 VLLM_FIREFLY)。

    fp8=强制开; 0=强制关; auto(默认)= 跟随 VLLM_FIREFLY (firefly 开→AR 开)。
    """
    v = envs.VLLM_FIREFLY_AR
    if v == "fp8":
        return True
    if v == "0":
        return False
    return envs.VLLM_FIREFLY == "1"  # auto


_cuda_mod = None
_cuda_load_attempted = False


def _load_cuda_mod():
    """懒加载 firefly_allreduce.cu (torch.utils.cpp_extension.load), 失败 None。

    首次 allreduce 时编译 (缓存到 torch extensions 目录), 不拖 vllm import。
    只编 sm_75 (T10)。
    """
    global _cuda_mod, _cuda_load_attempted
    if _cuda_load_attempted:
        return _cuda_mod
    _cuda_load_attempted = True
    try:
        cu_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "firefly_allreduce.cu"
        )
        if not os.path.exists(cu_path):
            logger.warning(
                "firefly_allreduce source not found at %s; fallback NCCL", cu_path
            )
            return None
        from torch.utils.cpp_extension import load as _load_ext

        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        _cuda_mod = _load_ext(
            name="firefly_allreduce_cuda",
            sources=[cu_path],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        logger.info("firefly_allreduce CUDA kernel loaded")
    except Exception as e:  # noqa: BLE001 - 编译/无 CUDA 环境回退 NCCL
        logger.warning("firefly_allreduce ext load failed, fallback NCCL: %s", e)
        _cuda_mod = None
    return _cuda_mod


def _unlink_shm(name: str) -> None:
    from multiprocessing.shared_memory import SharedMemory

    try:
        sb = SharedMemory(name=name, create=False)
        sb._mmap.close()
        sb.close()
        sb._mmap.unlink()
    except FileNotFoundError:
        pass


class FireflyAllReduce:
    """fp8 2-GPU allreduce (SHM backend)。仅 world_size=2 (TP2) 启用。

    与 CustomAllreduce 同接口: __init__ 设 disabled, should_firefly_ar 判定,
    all_reduce 执行, destroy 清理。
    """

    def __init__(self, rank_in_group: int, world_size: int, device,
                 shm_name: str):
        self.rank = rank_in_group
        self.world_size = world_size
        self.device = device
        self.disabled = True
        if world_size != 2 or not firefly_ar_active():
            return
        self._mod = _load_cuda_mod()
        if self._mod is None:
            return

        # data_half (fp8 bytes) = max_size / 2 (fp16 2B/elem -> fp8 1B/elem)
        self._max_size = envs.VLLM_FIREFLY_AR_MAX_SIZE
        self._data_half = self._max_size // 2
        total = 2 * self._data_half + 56

        from multiprocessing.shared_memory import SharedMemory

        # 建/挂 SHM: rank0 建 (先清旧), rank1 重试挂 (两 rank 同时起, 用重试兜底)
        if rank_in_group == 0:
            _unlink_shm(shm_name)
            time.sleep(0.3)
            self._shm = SharedMemory(name=shm_name, create=True, size=total)
        else:
            self._shm = None
            for _ in range(400):
                try:
                    self._shm = SharedMemory(name=shm_name, create=False)
                    break
                except FileNotFoundError:
                    time.sleep(0.05)
            if self._shm is None:
                logger.warning(
                    "firefly ar SHM attach timeout: %s; fallback NCCL", shm_name
                )
                return
        self._arr = (ctypes.c_char * total).from_buffer(self._shm.buf)
        self._host_ptr = ctypes.addressof(self._arr)
        r = _LIB.cudaHostRegister(self._host_ptr, total, 0)
        if r != 0:
            logger.warning(
                "firefly ar cudaHostRegister failed err=%d; fallback NCCL", r
            )
            self._teardown_shm()
            return

        # fp8 data scratch (device, max size; 实际 allreduce 用 n 字节子集)
        self._xq = torch.empty(self._data_half, dtype=torch.uint8, device=device)
        self._xq_peer = torch.empty(self._data_half, dtype=torch.uint8,
                                    device=device)
        # GPU state (16B): [+0]amax(f32) [+4]scale(f32) [+8]seq(u64); seq 初始 1
        #   (首次 flag 用 1, SHM flag 初始 0)。by-pointer, cudagraph replay 安全。
        self._scratch = torch.zeros(2, dtype=torch.int64, device=device)
        self._scratch[1] = 1
        self._base = self._host_ptr
        self._shm_name = shm_name
        self.disabled = False
        logger.info(
            "firefly_allreduce enabled rank=%d shm=%s %.1fMB",
            rank_in_group, shm_name, total / 1e6,
        )

    def should_firefly_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled or self.world_size != 2:
            return False
        if inp.dtype != torch.float16 or not inp.is_contiguous():
            return False
        if inp.numel() == 0 or inp.numel() * 2 > self._max_size:
            return False
        return True

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        # 全 GPU: round1 (全局 amax + SHM 交换 -> common scale) + round2 (quant +
        #   D2H + data flag + H2D + dequant+sum + barrier + bump seq) 都在 GPU
        #   stream, 无 CPU sync (.item()), 可被 cudagraph capture。
        out = torch.empty_like(inp)
        self._mod.firefly_ar_exchange(
            inp, self._xq, self._xq_peer, out, self._scratch, self._base,
            self._data_half, self.rank, inp.numel(),
        )
        return out

    def _teardown_shm(self) -> None:
        try:
            del self._arr
            self._shm.close()
        except Exception:  # noqa: BLE001
            pass

    def destroy(self) -> None:
        if self.disabled:
            return
        try:
            _LIB.cudaHostUnregister(self._host_ptr)
        except Exception:  # noqa: BLE001
            pass
        self._teardown_shm()
        if self.rank == 0:
            try:
                self._shm.unlink()
            except Exception:  # noqa: BLE001
                pass
        self.disabled = True
