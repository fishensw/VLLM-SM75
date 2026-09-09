"""Run in the image without GPUs: derived block sizes must not leak into launch."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.engine.core_client import MPClient


class RespawnCacheConfigTests(unittest.TestCase):
    def make_client(self):
        client = MPClient.__new__(MPClient)
        client.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=832, num_gpu_blocks=100),
            parallel_config=SimpleNamespace(data_parallel_size=1),
        )
        client._respawn_initial_block_size = 32
        client._respawn_addresses = object()
        client._respawn_executor_class = object()
        client._respawn_log_stats = False
        client.resources = SimpleNamespace()
        return client

    def test_launch_copy_restores_input_without_changing_frontend(self):
        client = self.make_client()
        config = client._make_respawn_config()
        self.assertEqual(config.cache_config.block_size, 32)
        self.assertEqual(config.cache_config.num_gpu_blocks, 0)
        self.assertEqual(client.vllm_config.cache_config.block_size, 832)
        self.assertEqual(client.vllm_config.cache_config.num_gpu_blocks, 100)
        self.assertIsNot(config.cache_config, client.vllm_config.cache_config)

    def test_repeated_launch_never_reuses_reported_block_size(self):
        client = self.make_client()
        observed = []

        @contextmanager
        def launch(config, *args):
            observed.append(config.cache_config.block_size)
            yield SimpleNamespace(coordinator=None, engine_manager=None)

        with patch("vllm.v1.engine.core_client.launch_core_engines", launch):
            for reported in (832, 1664, 832):
                client.vllm_config.cache_config.block_size = reported
                client.vllm_config.cache_config.num_gpu_blocks = 100
                client._respawn_launch()
                self.assertEqual(client.vllm_config.cache_config.block_size, reported)
                self.assertEqual(client.vllm_config.cache_config.num_gpu_blocks, 0)
        self.assertEqual(observed, [32, 32, 32])


if __name__ == "__main__":
    unittest.main()
