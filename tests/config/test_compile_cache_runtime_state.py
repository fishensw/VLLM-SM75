"""Run in the built image: runtime counters must not invalidate draft caches."""

import unittest

from vllm.config import CompilationConfig


class CompilationRuntimeStateTests(unittest.TestCase):
    def test_operator_statistics_do_not_change_cache_key(self):
        config = CompilationConfig()
        original = config.compute_hash()
        config.enabled_custom_ops.update({"rms_norm": 64, "rotary_embedding": 4})
        config.disabled_custom_ops.update({"silu_and_mul": 64})
        self.assertEqual(original, config.compute_hash())

    def test_operator_policy_still_changes_cache_key(self):
        config = CompilationConfig()
        original = config.compute_hash()
        config.custom_ops.append("+rms_norm")
        self.assertNotEqual(original, config.compute_hash())


if __name__ == "__main__":
    unittest.main()
