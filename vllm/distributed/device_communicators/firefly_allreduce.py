# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FireflyAllReduce: fp8 2-GPU allreduce (SHM backend, PHB/无 P2P 如 T10)。

TP2 每层 2 次 AllReduce([M,H] fp16 residual stream) 在 PCIe 瓶颈上量减半
(fp16 -> fp8), 缓解 27B prefill 通信主导 (PLAN-fp8-allreduce.md)。

传输: /dev/shm + cudaHostRegister (两进程 map 同一 tmpfs 区, 各自 register 成
pinned 设备可访问, 物理页经 tmpfs 共享)。POC 验证 vfio 下可行。

只支持 fp16 输入。world_size=2 (TP2) 走 2 卡互发 (SHM/P2P);
world_size 为 >=4 的 2 的幂 (TP4/8/...) 走 butterfly (SHM/P2P, 无 P2P 如 T10
走 SHM, 单一 shm_base host-mapped 区 + D2H/H2D memcpy)。

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


def firefly_ar_world_ok(world_size: int) -> bool:
    """FireflyAllReduce 支持的 world_size: 任意 2 的幂 (2/4/8/16/...)。

    2 卡走互发 (SHM/P2P); N>=4 走 butterfly (递归折半, log2(N) 整数轮, 故 N
    必为 2 的幂, 不限上限 —— 算法对任意 2 的幂成立, 实际上限由硬件定:
    每 rank IPC buffer = N*D 显存 + N 卡全互联 P2P (如 NVSwitch) 拓扑。
    """
    return world_size >= 2 and (world_size & (world_size - 1)) == 0


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
    """清 stale 段 (上一进程 pkill -9 未走 destroy unlink 残留)。

    必须用公共 API:
      - sb.close()  内部 release buf + close mmap + close fd (buf 不先 release 会
                    BufferError: cannot close exported pointers exist)。
      - sb.unlink() 真正的 shm_unlink。
    坑: mmap.mmap 没有 .unlink() 方法 (sm75 环境实测 hasattr==False), 用
      sb._mmap.unlink() 会 AttributeError 被吞 → 段根本没删, 下次 create=True
      (O_CREAT|O_EXCL) 撞 EEXIST 挡启动。曾踩。
    """
    from multiprocessing.shared_memory import SharedMemory

    try:
        sb = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    sb.close()
    try:
        sb.unlink()
    except Exception:  # noqa: BLE001
        pass


class FireflyAllReduce:
    """fp8 allreduce: 2 卡 (TP2) SHM/P2P, N 卡 (TP4/8) butterfly。

    world_size==2 走 2 卡互发 (无 P2P 走 SHM, 有 P2P 走 IPC 显存直读);
    world_size in {4,8} 走 P2P butterfly (递归折半, logN 轮, IPC 显存)。
    与 CustomAllreduce 同接口: __init__ 设 disabled, should_firefly_ar 判定,
    all_reduce 执行, destroy 清理。
    """

    def __init__(self, rank_in_group: int, world_size: int, device,
                 shm_name: str, group=None):
        self.rank = rank_in_group
        self.world_size = world_size
        self.device = device
        self.disabled = True
        self._backend = "shm"
        self._group = group
        self._bases_dev = None
        if not firefly_ar_world_ok(world_size) or not firefly_ar_active():
            return
        self._mod = _load_cuda_mod()
        if self._mod is None:
            return

        # data_half (fp8 bytes) = max_size / 2 (fp16 2B/elem -> fp8 1B/elem)
        self._max_size = envs.VLLM_FIREFLY_AR_MAX_SIZE
        self._data_half = self._max_size // 2
        self._ptrs = None

        if world_size >= 4:
            # N 卡 butterfly: auto 按 _can_p2p 选 (无 P2P 如 T10 → SHM), 或
            # VLLM_FIREFLY_AR_BACKEND 强制 shm/p2p。
            self._backend = self._select_backend_n(rank_in_group)
            if self._backend == "p2p" and not self._init_p2p_butterfly():
                self._backend = "shm"
            if self._backend == "shm":
                total = world_size * self._data_half + 28 * world_size
                if not self._init_shm_n(total, shm_name):
                    return
        else:  # world_size == 2: P2P 优先 (无 host bounce), 无 P2P 回 SHM
            total = 2 * self._data_half + 56
            self._backend = self._select_backend(rank_in_group)
            if self._backend == "p2p" and not self._init_p2p(total):
                self._backend = "shm"
            if self._backend == "shm" and not self._init_shm(total, shm_name):
                return

        # GPU state (16B): [+0]amax(f32) [+4]scale(f32) [+8]seq(u64); seq 初始 1
        #   (首次 flag 用 1, flag 初始 0)。by-pointer, cudagraph replay 读当前值。
        self._scratch = torch.zeros(2, dtype=torch.int64, device=self.device)
        self._scratch[1] = 1
        if world_size >= 4:
            # butterfly 运行部分和缓冲 (fp16, max M*H); 首轮前 all_reduce 拷入 x
            self._work = torch.empty(self._max_size // 2,
                                     dtype=torch.float16, device=self.device)
        self.disabled = False
        logger.info(
            "firefly_allreduce enabled rank=%d backend=%s world=%d",
            rank_in_group, self._backend, world_size,
        )

    def _select_backend(self, rank_in_group: int) -> str:
        """P2P 优先 (有 P2P 无 host bounce), 否则 SHM。VLLM_FIREFLY_AR_BACKEND
        可强制 p2p/shm; auto(默认) 运行时按 _can_p2p 选。"""
        pref = envs.VLLM_FIREFLY_AR_BACKEND
        if pref == "p2p":
            return "p2p"
        if pref == "shm":
            return "shm"
        # auto: 有 P2P 选 P2P (group 缺失无法交换 IPC handle, 回 SHM)
        if self._group is None:
            return "shm"
        try:
            from vllm.distributed.device_communicators.custom_all_reduce import (
                _can_p2p,
            )
            return "p2p" if _can_p2p(rank_in_group, self.world_size) else "shm"
        except Exception as e:  # noqa: BLE001 - P2P 检测失败回 SHM
            logger.debug("firefly ar P2P check failed, use SHM: %s", e)
            return "shm"

    def _init_p2p(self, total: int) -> bool:
        """P2P backend: IPC buffer 交换拿 own/peer 显存指针 (data/flag 全 device,
        无 host bounce)。复用 CustomAllreduce 的 IPC 分配+handle 交换机制。"""
        try:
            from vllm.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )
            # pointers[i] = 第 i 个 rank 的 buffer 显存指针 (本端 allocate, 对端
            #   open handle); own=pointers[rank], peer=pointers[1-rank] (world=2)。
            self._ptrs = CustomAllreduce.create_shared_buffer(
                total, group=self._group
            )
            self._own_base = self._ptrs[self.rank]
            self._peer_base = self._ptrs[1 - self.rank]
            self._buffer_total = total
            # cudaMalloc'd 显存 metadata 区是垃圾 → 清零防假 flag 命中 (56B)
            self._mod.firefly_ar_zero_meta(self._own_base, self._data_half, 2)
            for r in range(2):
                if r != self.rank:
                    self._mod.firefly_ar_zero_meta(
                        self._peer_base, self._data_half, 2
                    )
            torch.cuda.synchronize()
            return True
        except Exception as e:  # noqa: BLE001 - IPC 不可用回 SHM
            logger.warning(
                "firefly ar P2P init failed, fallback SHM: %s", e
            )
            return False

    def _select_backend_n(self, rank_in_group: int) -> str:
        """N 卡 (>=4) butterfly 后端: VLLM_FIREFLY_AR_BACKEND 优先, auto 按
        _can_p2p 选 (无 P2P 如 T10 PCIe → SHM, 有 P2P → p2p)。"""
        pref = envs.VLLM_FIREFLY_AR_BACKEND
        if pref == "p2p":
            return "p2p"
        if pref == "shm":
            return "shm"
        # auto: 有 P2P 选 P2P (group 缺失无法交换 IPC handle, 回 SHM)
        if self._group is None:
            return "shm"
        try:
            from vllm.distributed.device_communicators.custom_all_reduce import (
                _can_p2p,
            )
            return "p2p" if _can_p2p(rank_in_group, self.world_size) else "shm"
        except Exception as e:  # noqa: BLE001 - P2P 检测失败回 SHM
            logger.debug("firefly ar N-GPU P2P check failed, use SHM: %s", e)
            return "shm"

    def _init_p2p_butterfly(self) -> bool:
        """N 卡 (4/8) butterfly: 单 IPC buffer (N*data_half + 24 + 16N 字节,
        每 rank 一 buffer, 内含本 rank 的 data slot + per-rank flag 数组),
        交换 handle 拿全体 base → _bases_dev (device 数组)。"""
        try:
            from vllm.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            n = self.world_size
            total = n * self._data_half + 24 + 16 * n
            self._buffer_total = total
            self._ptrs = CustomAllreduce.create_shared_buffer(
                total, group=self._group
            )
            # bases_dev: device int64 数组, 供 ar_scale_exchange_n 读 (kernel 内
            # 不能解 host 指针), 含自己 (pointers[rank] = own 显存地址)。
            self._bases_dev = torch.tensor(
                self._ptrs, dtype=torch.int64, device=self.device
            )
            # cudaMalloc'd 显存 metadata 区 (amax/flag/barrier) 是垃圾 → 清零防
            #   假 flag 命中、dequant 读未写 data (对 own + 每个 peer 各一次)。
            for r in range(n):
                self._mod.firefly_ar_zero_meta(
                    self._ptrs[r], self._data_half, n
                )
            torch.cuda.synchronize()
            return True
        except Exception as e:  # noqa: BLE001 - IPC/P2P 不可用回退 NCCL
            logger.warning(
                "firefly ar butterfly init failed, fallback NCCL: %s", e
            )
            self._bases_dev = None
            return False

    def _init_shm_n(self, total: int, shm_name: str) -> bool:
        """N 卡 (>=4) SHM butterfly: 单一 shm_base (host-mapped pinned, 所有
        rank map 同一 tmpfs 区)。布局: [0, N*D) data slots, 之后 28N 字节
        metadata (amax/flag_scale/flag_data/barrier 各 N 槽)。rank0 建并清
        metadata, 其余 attach (重试保证 rank0 建完)。"""
        from multiprocessing.shared_memory import SharedMemory

        if self.rank == 0:
            # create=True 走 O_CREAT|O_EXCL: stale 段残留 → EEXIST, 重试
            # unlink+create 兜底 (同 2 卡 _init_shm)。
            for _attempt in range(10):
                _unlink_shm(shm_name)
                try:
                    self._shm = SharedMemory(
                        name=shm_name, create=True, size=total
                    )
                    break
                except FileExistsError:
                    time.sleep(0.3)
                    self._shm = None
            if self._shm is None:
                logger.warning(
                    "firefly ar SHM create timeout (stale %s); fallback NCCL",
                    shm_name,
                )
                return False
            # 清零 metadata 区 (tmpfs 新建段本已全 0, 显式清幂等稳妥; 只清
            #   末 28N 字节 metadata 即可, data 区无所谓——但 tmpfs 新建段
            #   全 0, 此处全清也可; 其余 rank attach 重试已保证在 rank0 建完后)
            n_meta = 28 * self.world_size
            meta_off = total - n_meta
            self._shm.buf[meta_off:total] = b"\x00" * n_meta
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
                return False
        self._arr = (ctypes.c_char * total).from_buffer(self._shm.buf)
        self._host_ptr = ctypes.addressof(self._arr)
        r = _LIB.cudaHostRegister(self._host_ptr, total, 0)
        if r != 0:
            logger.warning(
                "firefly ar cudaHostRegister failed err=%d; fallback NCCL", r
            )
            self._teardown_shm()
            return False

        # fp8 data scratch (device, max size; 实际 allreduce 用 n 字节子集)
        self._xq = torch.empty(self._data_half, dtype=torch.uint8,
                               device=self.device)
        self._xq_peer = torch.empty(self._data_half, dtype=torch.uint8,
                                    device=self.device)
        self._base = self._host_ptr
        self._shm_name = shm_name
        return True

    def _init_shm(self, total: int, shm_name: str) -> bool:
        """SHM backend: /dev/shm + cudaHostRegister (无 P2P, 如 T10 PHB)。"""
        from multiprocessing.shared_memory import SharedMemory

        # 建/挂 SHM: rank0 建 (先清旧), rank1 重试挂 (两 rank 同时起, 用重试兜底)
        if self.rank == 0:
            # create=True 走 O_CREAT|O_EXCL: stale 段残留 (上一进程 pkill -9 未走
            #   destroy unlink) 且 _unlink_shm 未及时清掉 → EEXIST。重试 unlink+create
            #   兜底 (rank1 并发 attach 不持有 name, shm_unlink 即刻生效)。
            for _attempt in range(10):
                _unlink_shm(shm_name)
                try:
                    self._shm = SharedMemory(
                        name=shm_name, create=True, size=total
                    )
                    break
                except FileExistsError:
                    time.sleep(0.3)
                    self._shm = None
            if self._shm is None:
                logger.warning(
                    "firefly ar SHM create timeout (stale %s); fallback NCCL",
                    shm_name,
                )
                return False
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
                return False
        self._arr = (ctypes.c_char * total).from_buffer(self._shm.buf)
        self._host_ptr = ctypes.addressof(self._arr)
        r = _LIB.cudaHostRegister(self._host_ptr, total, 0)
        if r != 0:
            logger.warning(
                "firefly ar cudaHostRegister failed err=%d; fallback NCCL", r
            )
            self._teardown_shm()
            return False

        # fp8 data scratch (device, max size; 实际 allreduce 用 n 字节子集)
        self._xq = torch.empty(self._data_half, dtype=torch.uint8,
                               device=self.device)
        self._xq_peer = torch.empty(self._data_half, dtype=torch.uint8,
                                    device=self.device)
        self._base = self._host_ptr
        self._shm_name = shm_name
        return True

    def should_firefly_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype != torch.float16 or not inp.is_contiguous():
            return False
        nbytes = inp.numel() * 2  # fp16 字节数
        # 下限: decode 小消息的 amax 扫描 + 多轮 flag spin 固定开销可能超过
        #   砍半省下的传输时间 → 反而更慢, 回退 NCCL。
        # 上限: 大消息超 shm/p2p buffer, 回退 NCCL。
        if nbytes < envs.VLLM_FIREFLY_AR_MIN_SIZE or nbytes > self._max_size:
            return False
        return True

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        # 全 GPU, 无 CPU sync (.item()), 可被 cudagraph capture。
        #   2-GPU SHM: round1 (amax + SHM 交换) + round2 (quant + D2H + flag
        #        + H2D + dequant+sum + barrier + bump seq)。
        #   2-GPU P2P: 同流程但 data/flag 全 device 显存 (quant 写 own_data,
        #        dequant 直读 peer_data), 无 D2H/H2D。
        #   N 卡 (4/8) butterfly: logN 轮, 每轮 amax-exchange(N) + quant +
        #        data flag + dequant(P2P 读 partner) + barrier + bump seq,
        #        部分和存 _work; 末轮拷入 out。data/flag 全 IPC 显存。
        n = inp.numel()
        out = torch.empty_like(inp)
        if self.world_size >= 4:
            self._work[:n] = inp.view(-1)
            if self._backend == "p2p":
                # bases_host = self._ptrs (host list[int], pybind 转 vector);
                #   bases_dev_ptr = device 数组首址, 仅作 ar_scale_exchange_n
                #   的 kernel 实参 (kernel 内解引用合法, host 不读)。
                self._mod.firefly_ar_butterfly(
                    inp, self._work, out, self._scratch, self._ptrs,
                    self._bases_dev.data_ptr(), self._data_half, self.rank, n,
                )
            else:  # shm: data/flag 全 host-mapped shm, D2H/H2D memcpy
                self._mod.firefly_ar_butterfly_shm(
                    inp, self._work, self._xq, self._xq_peer, out,
                    self._scratch, self._base, self._data_half, self.rank,
                    self.world_size, n,
                )
        elif self._backend == "p2p":
            self._mod.firefly_ar_exchange_p2p(
                inp, out, self._scratch, self._own_base, self._peer_base,
                self._data_half, self.rank, n,
            )
        else:
            self._mod.firefly_ar_exchange(
                inp, self._xq, self._xq_peer, out, self._scratch, self._base,
                self._data_half, self.rank, n,
            )
        return out

    def _teardown_shm(self) -> None:
        try:
            del self._arr
            self._shm.close()
        except Exception:  # noqa: BLE001
            pass

    def _teardown_p2p(self) -> None:
        try:
            from vllm.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )
            # 只释放本端 allocate 的 buffer (pointers[rank]); peer buffer 由
            # peer 释放。2-GPU P2P 与 N 卡 butterfly 共用 (都走 self._ptrs)。
            CustomAllreduce.free_shared_buffer(
                self._ptrs, group=self._group, rank=self.rank
            )
        except Exception:  # noqa: BLE001
            pass
        self._ptrs = None
        self._bases_dev = None

    def destroy(self) -> None:
        if self.disabled:
            return
        if self._backend == "p2p":
            self._teardown_p2p()
        else:  # shm: 2 卡 / N 卡共用 (都走 _host_ptr + _shm + rank0 unlink)
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
