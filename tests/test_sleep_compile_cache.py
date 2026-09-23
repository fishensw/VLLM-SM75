"""CPU-only regression checks for sleep settings in compilation cache keys.

B 方案: SM75 扩展 env 不再整文件覆盖上游 vllm/envs.py, 改为 envs_sm75.apply()
注入(把 EXTENSIONS 灌进 environment_variables + wrap compile_factors, 从 hash
pop 掉 INSTALL_IGNORED 的 auto-sleep 计时器)。本测试验证注入后的行为:

- 改 auto-sleep 计时/路径 → compile cache key 不变(被 wrapper pop)。
- 改真实 graph env(VLLM_USE_LAYERNAME, 上游字段) → cache key 仍失效。
- 固定 overlay 编译版本隔离旧缓存；同版本重复注入与 UI/休眠调整保持复用。

上游侧的 compile_factors 模拟成"ignored 不含 auto-sleep"(上游本不知道它们),
靠 apply() 的 wrapper pop 达成"不计入 hash", 比原来硬编码进 ignored 更贴近
真实注入路径。
"""

import hashlib
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_envs_sm75():
    """按文件加载工作区 vllm/envs_sm75.py(纯 stdlib, 无需 vllm/torch)。"""
    path = Path(__file__).resolve().parents[1] / "vllm" / "envs_sm75.py"
    spec = importlib.util.spec_from_file_location("sm75_envs_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SleepCompileCacheTests(unittest.TestCase):
    def setUp(self):
        envs_sm75 = _load_envs_sm75()

        # 构造上游风格 vllm.envs 模块: 只有上游 VLLM_USE_LAYERNAME getter;
        # compile_factors 的 ignored 不含 auto-sleep(模拟上游不感知它们)。
        fake_envs = types.ModuleType("vllm.envs")
        fake_envs.environment_variables = {
            "VLLM_USE_LAYERNAME": lambda: bool(
                int(os.getenv("VLLM_USE_LAYERNAME", "1"))
            ),
        }

        def _compile_factors(_envs=fake_envs):
            factors = {}
            for name, getter in _envs.environment_variables.items():
                factors[name] = getter()
            return factors

        fake_envs.compile_factors = _compile_factors

        # apply() 经 sys.modules["vllm.envs"] 取模块, 故 patch 后即可注入。
        patcher = patch.dict(sys.modules, {"vllm.envs": fake_envs})
        patcher.start()
        self.addCleanup(patcher.stop)
        envs_sm75.apply()

        # 裁剪到测试关心的项(保留 apply() 注入进来的真实 auto-sleep getter)。
        names = [
            "VLLM_AUTO_SLEEP_IDLE_TIMEOUT",
            "VLLM_AUTO_SLEEP_OFFLOAD_TARGET",
            "VLLM_AUTO_SLEEP_RELOAD_PATH",
            "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL",
            "VLLM_USE_LAYERNAME",
            "VLLM_MONITOR",
        ]
        fake_envs.environment_variables = {
            n: fake_envs.environment_variables[n] for n in names
        }
        self.envs = fake_envs
        self.envs_sm75 = envs_sm75

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

    def test_monitor_toggle_preserves_release_cache_without_disabling_ui(self):
        reference = self.cache_key(VLLM_MONITOR="0")
        self.assertEqual(reference, self.cache_key(VLLM_MONITOR="1"))
        with patch.dict(os.environ, {"VLLM_MONITOR": "1"}):
            legacy = self.envs.compile_factors.__wrapped__()
            legacy["VLLM_MONITOR"] = False
            for name in _load_envs_sm75().INSTALL_IGNORED:
                legacy.pop(name, None)
            legacy["_vllm_sm75_compile_revision"] = "0.1.6-1"
            self.assertEqual(legacy, self.envs.compile_factors())
            self.assertTrue(self.envs.environment_variables["VLLM_MONITOR"]())


    def test_release_revision_invalidates_the_legacy_fingerprint(self):
        current = self.envs.compile_factors()
        legacy = self.envs.compile_factors.__wrapped__()
        for name in self.envs_sm75.INSTALL_IGNORED:
            legacy.pop(name, None)
        legacy["VLLM_MONITOR"] = False
        self.assertEqual(current["_vllm_sm75_compile_revision"], "0.1.6-1")
        without_revision = {
            key: value for key, value in current.items()
            if key != "_vllm_sm75_compile_revision"
        }
        self.assertEqual(without_revision, legacy)

        def fingerprint(factors):
            return hashlib.sha256(
                json.dumps(factors, sort_keys=True).encode()
            ).hexdigest()

        self.assertNotEqual(fingerprint(legacy), fingerprint(current))
        self.assertNotIn(
            "_vllm_sm75_compile_revision", self.envs.environment_variables
        )

    def test_revision_is_fixed_and_repeated_apply_is_idempotent(self):
        # Restore the complete extension registry narrowed by setUp before
        # checking real repeated-install behavior.
        self.envs_sm75.apply()
        wrapper = self.envs.compile_factors
        reference = self.cache_key()
        with patch.dict(os.environ, {"_vllm_sm75_compile_revision": "override"}):
            self.envs_sm75.apply()
            self.envs_sm75.apply()
            self.assertIs(self.envs.compile_factors, wrapper)
            self.assertEqual(reference, self.cache_key())
            self.assertEqual(
                self.envs.compile_factors()["_vllm_sm75_compile_revision"],
                "0.1.6-1",
            )


if __name__ == "__main__":
    unittest.main()
