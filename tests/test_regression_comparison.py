import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('comparison', Path(__file__).parents[1] / 'tools/compare-regression.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparisonTests(unittest.TestCase):
    def samples(self):
        return [dict(inputTokens=8192, outputTokens=512, repeat=i, valid=True,
                     complete=True, retrievalPassed=True, cachedTokens=0, preemptions=0,
                     contract='test', model='model', promptTokenHash='prompt', tokenHash='output',
                     prefillTokensPerSecond=1000, decodeTokensPerSecond=100,
                     ttftSeconds=8.192, tpotSeconds=0.01) for i in range(1, 6)]

    def test_equal_and_small_variance(self):
        rows = self.samples()
        rows[0]['decodeTokensPerSecond'] = 90
        self.assertTrue(module.compare({'base': self.samples(), 'candidate': rows})['passed'])

    def test_regression_and_hash_failure_are_independent(self):
        for field, value in [('decodeTokensPerSecond', 90), ('tokenHash', 'different'),
                             ('cachedTokens', 1), ('promptTokenHash', 'other')]:
            with self.subTest(field=field):
                rows = self.samples()
                for row in rows:
                    row[field] = value
                self.assertFalse(module.compare({'base': self.samples(), 'candidate': rows})['passed'])

    def test_missing_and_duplicate_samples_fail(self):
        rows = self.samples()
        for candidate in ([], rows[:-1], rows + rows[:1]):
            self.assertFalse(module.compare({'base': self.samples(), 'candidate': candidate})['passed'])

    def test_cli_three_repeat_contract(self):
        import json
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ('base', 'candidate'):
                (root / (label + '.jsonl')).write_text(''.join(json.dumps(row) + "\n" for row in self.samples()[:3]))
            result = subprocess.run([sys.executable, str(Path(module.__file__)),
                str(root / 'base.jsonl'), str(root / 'candidate.jsonl'),
                '--repeats', '3', '--output', str(root / 'result.json')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads((root / 'result.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
