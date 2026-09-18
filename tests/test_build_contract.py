"""CPU-only checks for the unified public build context."""
import ast
import re
import unittest
import tempfile
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BuildContract(unittest.TestCase):
    def test_official_prebuilt_base(self):
        text = (ROOT / 'docker/Dockerfile').read_text(encoding='utf-8')
        self.assertIn('vllm/vllm-openai:v0.29.0-cu129@sha256:', text)
        script = (ROOT / 'docker/build.sh').read_text(encoding='utf-8')
        self.assertNotIn('git clone', script)
        self.assertNotIn('git init', script)
        self.assertNotIn('prepare_upstream', script)

    def test_official_entrypoint_inheritance(self):
        for name in ('Dockerfile', 'Dockerfile.fast'):
            text = (ROOT / 'docker' / name).read_text(encoding='utf-8')
            self.assertIsNone(re.search(r'^(ENTRYPOINT|CMD)\s', text, re.M))
        installer = (ROOT / 'docker/helpers/install_sm75_overlay.py').read_text(encoding='utf-8')
        for path in ('entrypoints/cli/main.py', 'entrypoints/cli/serve.py', 'entrypoints/serve/entry.py'):
            self.assertNotIn('"' + path + '"', installer)

    def test_installer(self):
        install = runpy.run_path(str(ROOT / 'docker/helpers/install_speculative.py'))['install']
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / 'site-packages/vllm'
            evidence = Path(tmp) / 'evidence/speculative-files.json'
            result = install(ROOT / 'docker/speculative', package, evidence)
            self.assertEqual(len(result), 8)
            self.assertTrue(evidence.is_file())
            self.assertTrue((package.parent / 'sm75_fa2_graph.py').is_file())

    def test_copied_inputs_exist(self):
        text = (ROOT / 'docker/Dockerfile').read_text(encoding='utf-8')
        text = text.replace('\\\n', ' ')
        for line in text.splitlines():
            if line.startswith('COPY ') and '--from=' not in line:
                for source in line.split()[1:-1]:
                    self.assertTrue((ROOT / source).exists(), source)

    def test_unified_final_target(self):
        text = (ROOT / 'docker/Dockerfile').read_text(encoding='utf-8')
        self.assertEqual(re.findall(r'^FROM .* AS (.*)$', text, re.M)[-1], 'final')
        self.assertIn('python3 /tmp/install_speculative.py', text)
        launch = (ROOT / 'docker/run.sh').read_text(encoding='utf-8')
        self.assertIn('image="vllm-sm75:v${VERSION}"', launch)
        self.assertIn('"$SCRIPT_DIR/VERSION"', launch)
        self.assertNotIn('image+=', launch)

    def test_source_syntax_and_activation(self):
        for path in (ROOT / 'docker/speculative').rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        text = (ROOT / 'docker/speculative/vllm/v1/worker/gpu/model_runner.py').read_text(encoding='utf-8')
        for hook in ('_sm75_fa2_graph.install()', '_sm75_gdn_meta.install()'):
            self.assertEqual(text.count(hook), 1)

    def test_pstate_assets_bundled(self):
        text = (ROOT / 'docker/Dockerfile').read_text(encoding='utf-8')
        self.assertIn('/usr/local/bin/nvidia-pstated', text)
        self.assertIn('NVIDIA_PSTATE_SHA256', text)
        self.assertIn('COPY docker/helpers/pstate-entrypoint.sh /opt/vllm-sm75/pstate-entrypoint.sh', text)
        self.assertIn('COPY docker/helpers/pstate-supervisor.sh /opt/vllm-sm75/pstate-supervisor.sh', text)

    def test_sm75_env_injection_registered(self):
        text = (ROOT / 'docker/Dockerfile').read_text(encoding='utf-8')
        self.assertIn('COPY vllm/envs_sm75.py /tmp/sm75-overlay/envs_sm75.py', text)
        self.assertIn('COPY vllm/v1/core/sched/scheduler_sm75.py /tmp/sm75-overlay/v1/core/sched/scheduler_sm75.py', text)
        self.assertNotIn('COPY vllm/envs.py ', text)

    def test_no_host_cache_copy(self):
        text = (ROOT / 'docker/Dockerfile').read_text(encoding='utf-8')
        for value in ('FROM local/', 'COPY flashinfer-cache', '/mnt/user/'):
            self.assertNotIn(value, text)
        self.assertIsNone(re.search(r'(?<!\d)10(?:\.\d{1,3}){3}(?!\d)', text))

    def test_one_current_recipe_and_helper_set(self):
        self.assertEqual(list((ROOT / 'docker').glob('Dockerfile*v0.*')), [])
        self.assertEqual(list((ROOT / 'docker').glob('BUILD-v*')), [])
        for recipe in ('Dockerfile', 'Dockerfile.fast'):
            text = (ROOT / 'docker' / recipe).read_text(encoding='utf-8')
            self.assertNotIn('COPY docker/install_', text)
            self.assertIn('COPY docker/helpers/install_sm75_overlay.py', text)


if __name__ == '__main__':
    unittest.main()
