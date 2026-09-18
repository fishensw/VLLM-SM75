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
        self.assertIn('none', args)

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
