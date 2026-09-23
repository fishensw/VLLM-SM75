"""Transport and evidence-contract checks; no model inference is performed."""
import http.server
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

spec = importlib.util.spec_from_file_location("sm75_serving_smoke", Path(__file__).parents[1] / "tools/validate-sm75-serving.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def exercise(self, fail_json=False, monitor=True):
        state = {"enabled": True, "toggles": [], "requests": [], "active": 0, "peak": 0}
        lock = threading.Lock()
        key = "synthetic-test-secret-not-real"

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value, status=200):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self.reply({"data": [{"id": "fixture-model"}]})
                if self.path == "/monitor/spec_decode":
                    if not monitor:
                        return self.reply({"error": "unavailable"}, 404)
                    return self.reply({"spec_configured": True, "enabled": state["enabled"], "pending": False})
                if self.path == "/metrics":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'vllm:spec_decode_num_accepted_tokens_total{model_name="fixture"} 12\nvllm:spec_decode_num_accepted_tokens_created 9999\n')
                    return
                return self.reply({"error": "missing"}, 404)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/monitor/spec_decode":
                    if not self.headers.get("x-api-key-hash"):
                        return self.reply({"error": "auth"}, 401)
                    state["enabled"] = payload["enabled"]
                    state["toggles"].append(payload["enabled"])
                    return self.reply({"ok": True})
                with lock:
                    state["requests"].append(payload)
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                try:
                    time.sleep(0.025)
                    if fail_json and "response_format" in payload:
                        return self.reply({"error": key}, 500)
                    text = json.dumps({"status": "ready", "count": 3}) if "response_format" in payload else ("enabled" if state["enabled"] else "disabled")
                    token = 100 if state["enabled"] else 101
                    usage = {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
                    if payload.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        for event in [
                            {"choices": [{"delta": {"content": text}, "token_ids": [token], "prompt_token_ids": [999], "finish_reason": "stop"}]},
                            {"choices": [], "usage": usage},
                        ]:
                            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                    else:
                        self.reply({"choices": [{"message": {"content": text}, "token_ids": [token], "finish_reason": "stop"}], "usage": usage})
                finally:
                    with lock:
                        state["active"] -= 1

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "result.json"
                client = smoke.Client(f"http://127.0.0.1:{server.server_port}/v1", key, 3)
                result = smoke.run(client, None, output, toggle_timeout=1)
                evidence = output.read_text(encoding="utf-8")
                self.assertNotIn(key, evidence)
                self.assertEqual(json.loads(evidence), result)
                return result, state
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_complete_contract_restores_toggle_and_records_greedy_differences(self):
        result, state = self.exercise()
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"], {"passed_cases": 12, "total_cases": 12, "greedy_differences": 3})
        self.assertEqual(state["toggles"], [False, True])
        self.assertTrue(state["enabled"])
        self.assertEqual(state["peak"], 4)
        self.assertEqual(result["cases"][3]["token_ids"], [100])
        self.assertEqual(result["metrics"]["before"]["aggregate"], {"vllm:spec_decode_num_accepted_tokens_total": 12})
        for payload in state["requests"]:
            self.assertEqual(payload["temperature"], 0)
            self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])

    def test_http_failure_is_recorded_and_credentials_are_redacted(self):
        result, state = self.exercise(fail_json=True)
        self.assertFalse(result["passed"])
        failed = [row for row in result["cases"] if not row["passed"]]
        self.assertEqual([row["case"] for row in failed], ["json_schema"])
        self.assertEqual(failed[0]["http_status"], 500)
        self.assertEqual(failed[0]["response"]["error"], "[REDACTED]")
        self.assertEqual(state["toggles"], [False, True])

    def test_monitor_absence_skips_comparison_but_runs_functional_checks(self):
        result, state = self.exercise(monitor=False)
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["total_cases"], 9)
        self.assertEqual(result["speculative_comparison"]["status"], "skipped")
        self.assertEqual(state["toggles"], [])

    def test_non_loopback_and_embedded_credentials_are_refused(self):
        for address in ["https://example.com", "http://127.0.0.1@evil.test", "http://key@localhost", "http://127.0.0.1/?key=secret"]:
            with self.assertRaises(ValueError):
                smoke.Client(address)
        self.assertEqual(smoke.loopback_base("http://[::1]:8000/v1"), "http://[::1]:8000")


if __name__ == "__main__":
    unittest.main()