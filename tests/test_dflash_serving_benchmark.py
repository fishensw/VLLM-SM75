"""Offline numerical/evidence checks only: never contacts a model server."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("long_serving", HERE.parent / "tools/benchmark-dflash-serving.py")
long = importlib.util.module_from_spec(spec)
spec.loader.exec_module(long)
smoke, smoke_path = long.load_smoke(HERE.parent)


def sample(drafts, drafted, accepted, other=0):
    data = {}
    for model, values in (("fixture", (drafts, drafted, accepted)), ("unrelated", (other, other * 7, other * 6))):
        for name, value in zip(long.COUNTERS.values(), values):
            data[name + '{model_name="' + model + '",engine="0"}'] = float(value)
    return {"series": data}


class ComparisonTests(unittest.TestCase):
    def test_delta_uses_window_exact_family_and_model(self):
        before = sample(100, 700, 400, 1000)
        after = sample(113, 786, 456, 100000)
        before["series"][long.POSITION + '{model_name="fixture",engine="0",position="0"}'] = 400.0
        after["series"][long.POSITION + '{model_name="fixture",engine="0",position="0"}'] = 456.0
        result = long.metric_delta(before, after, "fixture")
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["draft_tokens"], 86)
        self.assertEqual(result["accepted_tokens"], 56)
        self.assertAlmostEqual(result["token_acceptance_rate"], 56 / 86)
        self.assertAlmostEqual(result["mean_acceptance_length_including_verification_token"], 1 + 56 / 13)

    def test_resets_and_missing_series_cannot_be_high_acceptance(self):
        before = sample(100, 700, 400)
        after = sample(3, 21, 20)
        self.assertIsNone(long.metric_delta(before, after, "fixture")["token_acceptance_rate"])
        after = sample(101, 707, 407)
        del after["series"][long.COUNTERS["draft_tokens"] + '{model_name="fixture",engine="0"}']
        self.assertFalse(long.metric_delta(before, after, "fixture")["valid"])

    def test_six_fixed_serial_requests_and_weighted_summaries(self):
        class Fake(smoke.Client):
            def __init__(self):
                super().__init__("http://127.0.0.1:1", "synthetic-do-not-save")
                self.requests, self.counts = [], [100, 700, 400]

            def exchange(self, path, payload=None, **kwargs):
                base = {"path": path, "request": payload, "http_status": 200, "elapsed_seconds": 2}
                if path == "/v1/models":
                    return {**base, "response": {"data": [{"id": "fixture"}]}}
                if path == "/monitor/spec_decode":
                    return {**base, "response": {"spec_configured": True, "enabled": True}}
                if path == "/metrics":
                    metrics = sample(*self.counts)
                    return {**base, "text": "\n".join(f"{k} {v}" for k, v in metrics["series"].items())}
                self.requests.append(payload)
                index = len(self.requests)
                increments = [10, 70, 60] if index % 2 else [20, 140, 40]
                self.counts = [a + b for a, b in zip(self.counts, increments)]
                return {**base, "response": {
                    "choices": [{"message": {"content": "long fixture output " + self.key},
                                 "token_ids": list(range(256)), "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 60, "completion_tokens": 256, "total_tokens": 316}}}

        client = Fake()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(label="fixture", model=None, api_key_env="VLLM_API_KEY", output=Path(directory) / "results.json")
            with patch.object(long.time, "sleep", lambda _: None):
                result = long.run(smoke, smoke_path, client, args)
            self.assertTrue(result["passed"], result.get("fatal_error"))
            self.assertEqual(len(client.requests), 6)
            self.assertEqual([p["messages"][0]["content"] for p in client.requests], list(long.PROMPTS) * 2)
            self.assertTrue(all(p["max_tokens"] == 256 and p["temperature"] == 0 and p["seed"] == 0 and p["chat_template_kwargs"]["enable_thinking"] is False for p in client.requests))
            self.assertEqual(result["summary"]["completion_tokens"], 1536)
            self.assertEqual(result["summary"]["draft_tokens"], 630)
            self.assertEqual(result["summary"]["accepted_tokens"], 300)
            self.assertAlmostEqual(result["summary"]["token_acceptance_rate"], 300 / 630)
            self.assertNotIn(client.key, args.output.read_text(encoding="utf-8"))
            saved = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["cases"]), 6)


if __name__ == "__main__":
    unittest.main()
