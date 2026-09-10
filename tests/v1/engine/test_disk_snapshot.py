# SPDX-License-Identifier: Apache-2.0
"""Snapshot integrity and bounded-copy tests, runnable without CUDA/torch."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[3] / "vllm/device_allocator/disk_snapshot.py"
if not path.is_file():
    import vllm
    path = Path(vllm.__file__).resolve().parent / "device_allocator/disk_snapshot.py"
spec = importlib.util.spec_from_file_location("disk_snapshot_tested", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DiskSnapshot = module.DiskSnapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = bytearray(range(256)) * 47
        self.buffer = memoryview(bytearray(127))
        self.calls = []

    def read(self, address, chunk):
        self.calls.append(len(chunk))
        chunk[:] = self.data[address : address + len(chunk)]

    def save(self):
        return DiskSnapshot.save(
            self.temp.name,
            [(0, 1001, "weights"), (2000, 999, "draft")],
            self.buffer,
            self.read,
        )

    def test_two_regions_roundtrip_multiple_cycles(self):
        for _ in range(2):
            snapshot = self.save()
            restored = bytearray(len(self.data))

            def write(address, chunk, destination=restored):
                destination[address : address + len(chunk)] = chunk

            snapshot.restore_region(2000, self.buffer, write)
            snapshot.restore_region(0, self.buffer, write)
            self.assertEqual(restored[:1001], self.data[:1001])
            self.assertEqual(restored[2000:2999], self.data[2000:2999])
            self.assertLessEqual(max(self.calls), 127)
            self.assertTrue((snapshot.directory / "manifest.json").is_file())
            snapshot.remove()
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_failed_save_removes_partial_snapshot(self):
        def fail(address, chunk):
            if address > 128:
                raise OSError("simulated read/write failure")
            self.read(address, chunk)

        with self.assertRaises(OSError):
            DiskSnapshot.save(self.temp.name, [(0, 1001, "weights")], self.buffer, fail)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_corruption_is_detected_and_snapshot_retained(self):
        snapshot = self.save()
        with open(snapshot.directory / "payload.bin", "r+b") as file:
            file.write(b"corrupted")
        with self.assertRaisesRegex(OSError, "checksum"):
            snapshot.restore_region(0, self.buffer, lambda address, chunk: None)
        self.assertTrue(snapshot.directory.is_dir())

    def test_truncation_fails_before_any_device_write(self):
        snapshot = self.save()
        with open(snapshot.directory / "payload.bin", "r+b") as file:
            file.truncate(1)
        writes = []
        with self.assertRaisesRegex(OSError, "length"):
            snapshot.restore_region(
                0, self.buffer, lambda address, chunk: writes.append(address)
            )
        self.assertEqual(writes, [])


if __name__ == "__main__":
    unittest.main()
