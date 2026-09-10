# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory, durable byte snapshots for in-process disk sleep.

No torch dependency: the caller supplies one reusable transfer buffer and
callbacks. A snapshot is committed before the caller may release GPU memory.
It is deliberately not a process checkpoint: CUDA virtual addresses, graphs,
and communicators must remain alive in the original worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path


class DiskSnapshot:
    def __init__(self, directory: Path, entries: list[dict], total: int):
        self.directory = directory
        self.entries = entries
        self.total = total

    @staticmethod
    def _drop_cache(fd: int, offset: int, length: int) -> None:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)

    @classmethod
    def save(
        cls,
        root: str,
        regions: list[tuple[int, int, str]],
        buffer: memoryview,
        read: Callable[[int, memoryview], None],
    ) -> DiskSnapshot:
        """Persist all regions; no device allocation is modified on failure."""
        if not os.path.isabs(root):
            raise ValueError("Disk sleep directory must be absolute")
        if not buffer or buffer.readonly:
            raise ValueError("Disk sleep requires a nonempty writable buffer")
        Path(root).mkdir(parents=True, exist_ok=True)
        total = sum(size for _, size, _ in regions)
        if shutil.disk_usage(root).free < total + 64 * 1024**2:
            raise OSError("Insufficient free disk space for runtime model snapshot")
        directory = Path(tempfile.mkdtemp(prefix=f"worker-{os.getpid()}-", dir=root))
        entries = []
        try:
            with open(directory / "payload.bin", "w+b", buffering=0) as file:
                file.truncate(total)
                offset = 0
                flushed = 0
                for address, size, tag in regions:
                    if size <= 0:
                        raise ValueError("Invalid allocation size")
                    digest = hashlib.sha256()
                    start = offset
                    for relative in range(0, size, len(buffer)):
                        chunk = buffer[: min(len(buffer), size - relative)]
                        read(address + relative, chunk)
                        digest.update(chunk)
                        written = 0
                        while written < len(chunk):
                            count = file.write(chunk[written:])
                            if not count:
                                raise OSError("Short snapshot write")
                            written += count
                        offset += len(chunk)
                        # Bound dirty page cache as well as the transfer buffer.
                        if offset - flushed >= 64 * 1024**2:
                            os.fsync(file.fileno())
                            cls._drop_cache(file.fileno(), flushed, offset - flushed)
                            flushed = offset
                    entries.append(
                        {
                            "address": address,
                            "size": size,
                            "tag": tag,
                            "offset": start,
                            "sha256": digest.hexdigest(),
                        }
                    )
                os.fsync(file.fileno())
                cls._drop_cache(file.fileno(), 0, total)
            with open(directory / "manifest.json", "x", encoding="utf-8") as file:
                json.dump(
                    {
                        "version": 1,
                        "pid": os.getpid(),
                        "total": total,
                        "entries": entries,
                    },
                    file,
                )
                file.flush()
                os.fsync(file.fileno())
            if os.name == "posix":
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            return cls(directory, entries, total)
        except BaseException:
            shutil.rmtree(directory)
            raise

    def restore_region(
        self,
        address: int,
        buffer: memoryview,
        write: Callable[[int, memoryview], None],
    ) -> None:
        entry = next(item for item in self.entries if item["address"] == address)
        path = self.directory / "payload.bin"
        if path.stat().st_size != self.total:
            raise OSError("Disk snapshot length mismatch; refusing to resume")
        digest = hashlib.sha256()
        with open(path, "rb", buffering=0) as file:
            file.seek(entry["offset"])
            for relative in range(0, entry["size"], len(buffer)):
                chunk = buffer[: min(len(buffer), entry["size"] - relative)]
                count = 0
                while count < len(chunk):
                    got = file.readinto(chunk[count:])
                    if not got:
                        raise OSError("Truncated disk snapshot")
                    count += got
                digest.update(chunk)
                write(address + relative, chunk)
                self._drop_cache(file.fileno(), entry["offset"] + relative, len(chunk))
        if digest.hexdigest() != entry["sha256"]:
            raise OSError("Disk snapshot checksum mismatch; refusing to resume")

    def remove(self) -> None:
        shutil.rmtree(self.directory)
