"""Validate evidence collection; these are not model-quality tests."""
import http.server
import importlib.util
import json
from pathlib import Path
import threading
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("long_context", Path(__file__).parents[1] / "tools/validate-long-context.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CollectorTests(unittest.TestCase):
    def test_baselines_reject_failed_or_duplicate_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.jsonl"
            case = {"inputTokens": 8192, "repeat": 1, "passed": True}
            path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            self.assertEqual(module.read_baselines([path])[(8192, 1)], case)
            with self.assertRaises(ValueError):
                module.read_baselines([path, path])
            case["passed"] = False
            path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                module.read_baselines([path])

    def test_counter_samples_exclude_created_and_unrelated_series(self):
        result = module.parse_metrics('''# TYPE vllm:kv_offload_load_bytes_total counter
vllm:kv_offload_load_bytes_total{worker="0"} 1.5e3
vllm:kv_offload_load_bytes_total{worker="1"} 250
vllm:kv_offload_load_bytes_created{worker="0"} 9999999
vllm:kv_offload_allocation_failure_total 2
vllm:kv_offload_cpu_allocation_size_count 3
vllm:kv_offload_cpu_allocation_size_sum 7
vllm:kv_offload_cpu_cache_usage_perc 0
other_metric 123
''')
        self.assertEqual(result, {
            "vllm:kv_offload_load_bytes_total": 1750,
            "vllm:kv_offload_allocation_failure_total": 2,
            "vllm:kv_offload_cpu_allocation_size_count": 3,
            "vllm:kv_offload_cpu_allocation_size_sum": 7,
            "vllm:kv_offload_cpu_cache_usage_perc": 0,
        })

    def test_stream_collects_generated_ids_not_prompt_ids(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            declared_count = 3
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in (
                    {"choices": [{"text": "", "prompt_token_ids": [7, 8], "token_ids": []}]},
                    {"choices": [{"text": "O", "token_ids": [100, 101]}]},
                    {"choices": [{"text": "K", "token_ids": [102], "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": self.declared_count}},
                ):
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = module.Client(f"http://127.0.0.1:{server.server_port}", "synthetic", "model")
            result = client.generate([7, 8], 16, 5, 0)
            self.assertEqual(result["text"], "OK")
            self.assertEqual(result["tokenIds"], [100, 101, 102])
            self.assertEqual(result["tokenHash"], module.token_hash([100, 101, 102]))
            self.assertEqual(result["finishReason"], "stop")
            Handler.declared_count = 4
            with self.assertRaisesRegex(RuntimeError, "token-ID count"):
                client.generate([7, 8], 16, 5, 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_low_memory_refuses_before_any_network_request(self):
        with patch.object(module, "available_memory", return_value=1):
            client = module.Client("http://127.0.0.1:1", "synthetic", "model")
            with self.assertRaisesRegex(RuntimeError, "host-memory-floor"):
                client.generate([7], 16, 5, 100)


if __name__ == "__main__":
    unittest.main()
