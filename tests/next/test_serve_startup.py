"""Check Next API environment inheritance without importing vLLM or using GPUs."""
import json
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class ServeStartup(unittest.TestCase):
    def test_nccl_default_and_explicit_override_reach_exec(self):
        for settings, expected in [({}, "SYS"), ({"NCCL_P2P_LEVEL": "PIX"}, "PIX")]:
            with self.subTest(settings=settings), patch.dict(os.environ, settings, clear=True):
                observed = {}
                def execute(binary, args):
                    observed.update(binary=binary, args=args, level=os.environ["NCCL_P2P_LEVEL"])
                with patch.object(Path, "read_text", return_value=json.dumps({"args": ["--model", "test-model"]})), \
                     patch.object(os, "execv", side_effect=execute), \
                     patch.object(sys, "argv", ["serve.py", "--port", "8000"]):
                    runpy.run_path(str(ROOT / "serve.py"), run_name="__main__")
                self.assertEqual(observed["level"], expected)
                self.assertEqual(observed["args"][-4:], ["--model", "test-model", "--port", "8000"])


if __name__ == "__main__":
    unittest.main()
