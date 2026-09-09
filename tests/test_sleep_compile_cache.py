"""CPU-only regression checks for sleep settings in compilation cache keys."""

import hashlib
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class SleepCompileCacheTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "vllm" / "envs.py"
        spec = importlib.util.spec_from_file_location("sm75_cache_test_envs", path)
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)
        # Exercise the real getters and compile_factors without importing Torch.
        names = [
            "VLLM_AUTO_SLEEP_IDLE_TIMEOUT",
            "VLLM_AUTO_SLEEP_OFFLOAD_TARGET",
            "VLLM_AUTO_SLEEP_RELOAD_PATH",
            "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL",
            "VLLM_USE_LAYERNAME",
        ]
        self.envs.environment_variables = {
            name: self.envs.environment_variables[name] for name in names
        }
        utils = types.ModuleType("vllm.config.utils")
        # These selected getters return only JSON primitives.
        utils.normalize_value = lambda value: value
        self.mock_utils = patch.dict(sys.modules, {"vllm.config.utils": utils})
        self.mock_utils.start()
        self.addCleanup(self.mock_utils.stop)

    def cache_key(self, **settings):
        settings = {
            "VLLM_AUTO_SLEEP_IDLE_TIMEOUT": "1",
            "VLLM_AUTO_SLEEP_OFFLOAD_TARGET": "exit",
            "VLLM_AUTO_SLEEP_RELOAD_PATH": "/models/checkpoint",
            "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL": "600",
            "VLLM_USE_LAYERNAME": "1",
            **settings,
        }
        with patch.dict(os.environ, settings):
            factors = self.envs.compile_factors()
        return hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()

    def test_sleep_policy_changes_preserve_cache_key(self):
        reference = self.cache_key(
            VLLM_AUTO_SLEEP_IDLE_TIMEOUT="1",
            VLLM_AUTO_SLEEP_OFFLOAD_TARGET="exit",
            VLLM_AUTO_SLEEP_RELOAD_PATH="/models/checkpoint",
            VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL="600",
        )
        for settings in [
            {"VLLM_AUTO_SLEEP_IDLE_TIMEOUT": "30"},
            {"VLLM_AUTO_SLEEP_IDLE_TIMEOUT": "0"},
            {"VLLM_AUTO_SLEEP_OFFLOAD_TARGET": "reload"},
            {"VLLM_AUTO_SLEEP_RELOAD_PATH": "/another/checkpoint"},
            {"VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL": "0"},
        ]:
            with self.subTest(settings=settings):
                self.assertEqual(reference, self.cache_key(**settings))

    def test_graph_environment_still_invalidates_cache(self):
        self.assertNotEqual(
            self.cache_key(VLLM_USE_LAYERNAME="0"),
            self.cache_key(VLLM_USE_LAYERNAME="1"),
        )


if __name__ == "__main__":
    unittest.main()
