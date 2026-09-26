"""Exercise public command construction without Docker, GPUs or real credentials."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash') or ('C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else None)


@unittest.skipUnless(BASH and Path(BASH).exists(), 'Bash required')
class ReleaseScripts(unittest.TestCase):
    def invoke(self, script, **settings):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # Bash printf emits one argument per line; no shell eval is used.
            fake = directory / 'docker'
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$COMMAND_LOG"\n', encoding='utf-8')
            fake.chmod(0o755)
            logfile = directory / 'calls'
            env = {**os.environ, 'PATH': str(directory) + os.pathsep + os.environ['PATH'],
                   'COMMAND_LOG': logfile.as_posix(), 'SOURCE_REVISION': 'test-source',
                   'VLLM_API_KEY': 'synthetic-test-key', 'VLLM_SM75_CACHE_ROOT': '/tmp/sm75-test-cache',
                   **settings}
            env = {key: value for key, value in env.items() if value is not None}
            result = subprocess.run([BASH, str(ROOT / script)], env=env, cwd=ROOT,
                                    text=True, capture_output=True)
            calls = logfile.read_text().splitlines() if logfile.exists() else []
            return result, calls

    def test_standard_and_ultra_build_tags_share_version(self):
        version = (ROOT / 'docker/VERSION').read_text().strip()
        for edition, suffix in [('standard', ''), ('ultra', '-ultra')]:
            result, args = self.invoke('docker/build.sh', EDITION=edition, BUILD_MODE='full')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('vllm-sm75:v' + version + suffix, args)

    def test_fast_and_ui_are_explicit_modes(self):
        result, args = self.invoke('docker/build.sh', EDITION='standard', BUILD_MODE='fast')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any(a.endswith('/docker/Dockerfile.fast') for a in args))
        result, args = self.invoke('docker/build.sh', EDITION='ultra', BUILD_MODE='ui', RUNTIME_IMAGE='audited-runtime')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('BASE_IMAGE=audited-runtime', args)
        self.assertIn('--memory', args)
        self.assertEqual(args[args.index('--network') + 1], 'default')

    def test_lmcache_build_opt_in_reaches_both_ultra_recipes(self):
        for mode in ('full', 'ui'):
            result, args = self.invoke('docker/build.sh', EDITION='ultra', BUILD_MODE=mode,
                                      RUNTIME_IMAGE='audited-runtime', INSTALL_LMCACHE='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('INSTALL_LMCACHE=1', args)

    def test_context_template_is_forwarded_and_zero_cpu_disables_connector(self):
        override = json.dumps({'dtype': 'float16', 'text_config': {'rope_parameters':
            {'rope_type': 'yarn', 'factor': 4, 'original_max_position_embeddings': 262144},
            'max_position_embeddings': 1048576}})
        result, args = self.invoke('docker/run.sh', POWER_MODE='sleep',
            AUTO_SLEEP_IDLE_TIMEOUT='0', CPU_KV_GIB='0', GPU_KV_BYTES='4294967296',
            MAX_MODEL_LEN='1048576', MAX_NUM_SEQS='1', MAX_NUM_BATCHED_TOKENS='4096',
            HF_OVERRIDES=override)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--kv-transfer-config', args)
        for flag, value in [('--max-model-len','1048576'), ('--max-num-seqs','1'),
                            ('--max-num-batched-tokens','4096'),
                            ('--kv-cache-memory-bytes','4294967296'), ('--hf-overrides',override)]:
            self.assertEqual(args[args.index(flag)+1], value)
        result, args = self.invoke('docker/run.sh', POWER_MODE='sleep', CPU_KV_GIB='2')
        self.assertEqual(result.returncode, 0, result.stderr)
        kv = json.loads(args[args.index('--kv-transfer-config')+1])
        self.assertEqual(kv['kv_connector_extra_config']['cpu_bytes_to_use'], 2 * 1024**3)

    def test_nccl_default_and_override_reach_launchers(self):
        for script, edition in [('docker/run.sh', 'standard'),
                                ('docker/run.sh', 'ultra'),
                                ('docker/helpers/run-ultra.sh', 'ultra')]:
            for value, expected in [(None, 'SYS'), ('PIX', 'PIX')]:
                with self.subTest(script=script, edition=edition, value=value):
                    result, args = self.invoke(script, EDITION=edition,
                        POWER_MODE='sleep', AUTO_SLEEP_IDLE_TIMEOUT='0',
                        ULTRA_DATA_ROOT='/tmp/sm75-test-ultra', MODEL_ROOT='/tmp',
                        NCCL_P2P_LEVEL=value)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(args.count('NCCL_P2P_LEVEL=' + expected), 1)

    def test_next_launchers_preserve_nccl_overrides(self):
        for script in ('scripts/run-next.sh', 'scripts/run-next-api.sh'):
            for value, expected in [(None, 'SYS'), ('PIX', 'PIX')]:
                with self.subTest(script=script, value=value):
                    result, args = self.invoke(script, MODEL_ROOT='/tmp',
                        CACHE_ROOT='/tmp/sm75-next-test-cache',
                        ULTRA_DATA_ROOT='/tmp/sm75-next-test-panel',
                        NCCL_P2P_LEVEL=value)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(args.count('NCCL_P2P_LEVEL=' + expected), 1)

    def test_invalid_mode_cannot_call_docker(self):
        result, args = self.invoke('docker/build.sh', EDITION='unknown')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(args, [])

    def test_standard_runtime_keeps_optimization_contract(self):
        result, args = self.invoke('docker/run.sh', EDITION='standard', POWER_MODE='sleep',
                                  VARIANT='base', FORMAT='fp8', IMAGE='test-image')
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ['test-image', 'VLLM_FIREFLY=1', 'VLLM_FIREFLY_AR=auto',
                      '--async-scheduling', 'flashqla_sm75', '--enable-prefix-caching']:
            self.assertIn(value, args)
        self.assertEqual(args[args.index('--auto-sleep-idle-timeout') + 1], '30')
        kv = json.loads(args[args.index('--kv-transfer-config') + 1])
        self.assertEqual(kv['kv_connector_extra_config']['cpu_bytes_to_use'], 8589934592)

    def test_marlin_reduction_setting_reaches_container(self):
        for value in ('0', '1'):
            result, args = self.invoke('docker/run.sh', POWER_MODE='sleep',
                AUTO_SLEEP_IDLE_TIMEOUT='0', VLLM_MARLIN_USE_ATOMIC_ADD=value)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('VLLM_MARLIN_USE_ATOMIC_ADD=' + value, args)

    def test_ultra_dispatch_does_not_require_web_token_or_stop_services(self):
        result, args = self.invoke('docker/run.sh', EDITION='ultra', ULTRA_DATA_ROOT='/tmp/sm75-test-ultra',
                                  MODEL_ROOT='/tmp', VLLM_API_KEY='', IMAGE='test-ultra')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('test-ultra', args)
        self.assertIn('SM75_CONSOLE_ROOT=/data', args)
        self.assertIn('no-new-privileges', args)
        self.assertNotIn('stop', args)
        self.assertNotIn('--privileged', args)


if __name__ == '__main__':
    unittest.main()
