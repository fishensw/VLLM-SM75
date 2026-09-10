"""CPU-only regression checks for sleep settings in compilation cache keys.

B 方案: SM75 扩展 env 不再整文件覆盖上游 vllm/envs.py, 改为 envs_sm75.apply()
注入(把 EXTENSIONS 灌进 environment_variables + wrap compile_factors, 从 hash
pop 掉 INSTALL_IGNORED 的 auto-sleep 计时器)。本测试验证注入后的行为:

- 改 auto-sleep 计时/路径 → compile cache key 不变(被 wrapper pop)。
- 改真实 graph env(VLLM_USE_LAYERNAME, 上游字段) → cache key 仍失效。

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
        ]
        fake_envs.environment_variables = {
            n: fake_envs.environment_variables[n] for n in names
        }
        self.envs = fake_envs

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
