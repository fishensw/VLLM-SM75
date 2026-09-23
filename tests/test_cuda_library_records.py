"""Detect a CUDA 13 file overwriting a still-installed CUDA 12 distribution."""
import base64
import hashlib
import importlib.metadata as metadata
import runpy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

verify = runpy.run_path(str(Path(__file__).resolve().parents[1] / "docker/helpers/verify_cuda_libraries.py"))["verify_distribution_libraries"]


class CUDALibraryRecordTests(unittest.TestCase):
    def fixture(self, directory, actual=b"cuda12-library"):
        root = Path(directory)
        record = metadata.PackagePath("libnccl.so.2")
        expected = base64.urlsafe_b64encode(hashlib.sha256(b"cuda12-library").digest()).rstrip(b"=").decode()
        record.hash = metadata.FileHash("sha256=" + expected)
        (root / record).write_bytes(actual)
        return SimpleNamespace(files=[record], locate_file=lambda item: root / item)

    def test_original_wheel_bytes_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = verify(self.fixture(directory))
            self.assertEqual(result[0]["sha256"], hashlib.sha256(b"cuda12-library").hexdigest())

    def test_same_version_metadata_cannot_hide_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "differs from installed wheel"):
                verify(self.fixture(directory, b"cuda13-library"))

    def test_missing_shared_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            distribution = self.fixture(directory)
            (Path(directory) / "libnccl.so.2").unlink()
            with self.assertRaises(FileNotFoundError):
                verify(distribution)


if __name__ == "__main__":
    unittest.main()
