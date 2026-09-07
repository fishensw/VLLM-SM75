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
        text = (ROOT / 'docker/Dockerfile.vllm-sm75-v0.1.3').read_text(encoding='utf-8')
        self.assertIn('vllm/vllm-openai:v0.28.0-cu129@sha256:', text)
        script = (ROOT / 'docker/build-v0.1.3.sh').read_text(encoding='utf-8')
        self.assertNotIn('git clone', script)
        self.assertNotIn('git init', script)
        self.assertNotIn('prepare_upstream', script)

    def test_installer(self):
        install = runpy.run_path(str(ROOT / 'docker/install_speculative.py'))['install']
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / 'site-packages/vllm'
            evidence = Path(tmp) / 'evidence/speculative-files.json'
            result = install(ROOT / 'docker/speculative', package, evidence)
            self.assertEqual(len(result), 6)
            self.assertTrue(evidence.is_file())
            self.assertTrue((package.parent / 'sm75_fa2_graph.py').is_file())

    def test_copied_inputs_exist(self):
        text = (ROOT / 'docker/Dockerfile.vllm-sm75-v0.1.3').read_text(encoding='utf-8')
        text = text.replace('\\\n', ' ')
        for line in text.splitlines():
            if line.startswith('COPY ') and '--from=' not in line:
                for source in line.split()[1:-1]:
                    self.assertTrue((ROOT / source).exists(), source)

    def test_unified_final_target(self):
        text = (ROOT / 'docker/Dockerfile.vllm-sm75-v0.1.3').read_text(encoding='utf-8')
        self.assertEqual(re.findall(r'^FROM .* AS (.*)$', text, re.M)[-1], 'final')
        self.assertIn('python3 /tmp/install_speculative.py', text)
        launch = (ROOT / 'docker/run-v0.1.3.sh').read_text(encoding='utf-8')
        self.assertIn('image=vllm-sm75:v0.1.3', launch)
        self.assertNotIn('image+=', launch)

    def test_source_syntax_and_activation(self):
        for path in (ROOT / 'docker/speculative').rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        text = (ROOT / 'docker/speculative/vllm/v1/worker/gpu/model_runner.py').read_text(encoding='utf-8')
        for hook in ('_sm75_fa2_graph.install()', '_sm75_gdn_meta.install()'):
            self.assertEqual(text.count(hook), 1)

    def test_no_host_cache_copy(self):
        text = (ROOT / 'docker/Dockerfile.vllm-sm75-v0.1.3').read_text(encoding='utf-8')
        for value in ('FROM local/', 'COPY flashinfer-cache', '/mnt/user/'):
            self.assertNotIn(value, text)
        self.assertIsNone(re.search(r'(?<!\d)10(?:\.\d{1,3}){3}(?!\d)', text))


if __name__ == '__main__':
    unittest.main()
