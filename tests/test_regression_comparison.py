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


if __name__ == '__main__':
    unittest.main()
