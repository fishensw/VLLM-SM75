# SPDX-License-Identifier: Apache-2.0
"""Disk-backed CuMem sleep: persist runtime bytes without a full CPU copy."""

from __future__ import annotations

import ctypes
import gc
import os

import torch
from vllm.device_allocator.sleep_mode_backend import CuMemBackend
from vllm.logger import init_logger

from vllm.device_allocator import cumem
from vllm.device_allocator.disk_snapshot import DiskSnapshot

logger = init_logger(__name__)


def _require_disk(path: str) -> None:
    """Reject RAM-backed paths; /dev/shm would recreate the original OOM."""
    if not path or not os.path.isabs(path):
        raise ValueError("Set --auto-sleep-disk-path to an absolute disk directory")
    os.makedirs(path, exist_ok=True)
    # Linux statfs starts with a native long f_type. Reserve enough space for
    # the entire struct without relying on a libc-specific Python definition.
    result = ctypes.create_string_buffer(256)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.statfs(os.fsencode(path), ctypes.byref(result)) != 0:
        raise OSError(ctypes.get_errno(), "statfs failed", path)
    filesystem = ctypes.c_long.from_buffer(result).value
    if filesystem in (0x01021994, 0x858458F6):  # tmpfs, ramfs
        raise ValueError("Disk sleep path is RAM-backed; mount a disk directory")


class DiskSleepBackend(CuMemBackend):
    """Preserve runtime model allocations and virtual addresses in one worker.

    KV contents are invalidated by the normal EngineCore sleep contract;
    their capacity/settings do not change. CUDA contexts/graphs stay alive.
    This backend does not support restoring after process/container restart.
    """

    def __init__(self):
        super().__init__()
        self.root = os.environ.get("VLLM_AUTO_SLEEP_DISK_PATH", "")
        self.snapshot: DiskSnapshot | None = None

    @classmethod
    def is_supported(cls) -> bool:
        from vllm.platforms import current_platform

        return current_platform.is_cuda() and cumem.cumem_available

    def _buffer(self):
        tensor = torch.empty(
            16 * 1024**2, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        return tensor, memoryview(tensor.numpy())

    def prepare(self) -> None:
        """Persist only; all TP ranks must finish this before any rank sleeps."""
        if self.snapshot is not None:
            return
        _require_disk(self.root)
        allocator = cumem.CuMemAllocator.get_instance()
        torch.cuda.synchronize()
        tensor, buffer = self._buffer()
        cpu_ptr = tensor.data_ptr()
        regions = [
            (ptr, data.handle[1], data.tag)
            for ptr, data in allocator.pointer_to_data.items()
            if data.tag != "kv_cache" and not data.is_asleep
        ]

        def read(address, chunk):
            cumem.libcudart.cudaMemcpy(cpu_ptr, address, len(chunk))

        # Commit every model allocation before releasing even the first one.
        self.snapshot = DiskSnapshot.save(self.root, regions, buffer, read)
        logger.info(
            "disk-sleep: committed %d runtime bytes to %s",
            self.snapshot.total,
            self.snapshot.directory,
        )
        del buffer, tensor

    def cancel_preparation(self) -> None:
        if self._state != "RUNNING":
            raise RuntimeError("Cannot cancel a snapshot after device unmapping")
        snapshot, self.snapshot = self.snapshot, None
        if snapshot is not None:
            snapshot.remove()

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("Disk sleep preserves runtime weights; use level 1")
        if self._state == "SUSPENDED":
            return
        self.prepare()
        allocator = cumem.CuMemAllocator.get_instance()
        try:
            for data in allocator.pointer_to_data.values():
                if not data.is_asleep:
                    cumem.unmap_and_release(data.handle)
                    data.is_asleep = True
        except Exception:
            # Keep the snapshot; the controller rolls all TP ranks back.
            self._state = "SUSPENDED"
            raise
        self._state = "SUSPENDED"
        gc.collect()
        torch.cuda.empty_cache()

    def resume(self, tags: list[str] | None = None) -> None:
        allocator = cumem.CuMemAllocator.get_instance()
        if self.snapshot is None:
            return  # Also supports rollback on a rank whose save failed.
        self._state = "RESUMING"
        tensor, buffer = self._buffer()
        cpu_ptr = tensor.data_ptr()
        saved = {item["address"] for item in self.snapshot.entries}

        def write(address, chunk):
            cumem.libcudart.cudaMemcpy(address, cpu_ptr, len(chunk))

        try:
            for ptr, data in allocator.pointer_to_data.items():
                if not data.is_asleep or (tags is not None and data.tag not in tags):
                    continue
                cumem.create_and_map(data.handle)
                try:
                    if ptr in saved:
                        self.snapshot.restore_region(ptr, buffer, write)
                except Exception:
                    cumem.unmap_and_release(data.handle)
                    raise
                data.is_asleep = False
            torch.cuda.synchronize()
        except Exception:
            self._state = "SUSPENDED"
            raise
        finally:
            del buffer, tensor
        if any(data.is_asleep for data in allocator.pointer_to_data.values()):
            self._state = "SUSPENDED"
            return
        logger.info(
            "disk-sleep: restored and verified %d runtime bytes from %s",
            self.snapshot.total,
            self.snapshot.directory,
        )
        try:
            self.snapshot.remove()
        except OSError:
            # Device data has already been verified. A cleanup failure must
            # not strand a fully restored model in RESUMING.
            logger.exception("disk-sleep: could not remove restored snapshot")
        self.snapshot = None
        self._state = "RUNNING"
