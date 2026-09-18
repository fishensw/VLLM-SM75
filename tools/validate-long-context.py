"""Bounded, synthetic long-context functional validation against an owned engine.

Uses exact token-ID inputs, three-position retrieval, natural stop, output token
hashes, streamed timing, and real offload counters. Not a performance benchmark.
The key stays in memory; every completed/failed case is appended and fsynced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
import urllib.error
import urllib.request


METRICS = {
    "vllm:kv_offload_load_bytes_total", "vllm:kv_offload_store_bytes_total",
    "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
    "vllm:num_preemptions_total", "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running", "vllm:num_requests_waiting",
}


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def read_baselines(paths):
    records = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = record["inputTokens"], record["repeat"]
            if not record.get("passed") or key in records:
                raise ValueError("Baseline must contain unique, passed cases only")
            records[key] = record
    return records


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(?:\{.*\})?\s+([^\s]+)", line)
        if match and match[1] in METRICS:
            result[match[1]] = result.get(match[1], 0) + float(match[2])
    return result


def available_memory():
    try:
        match = re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        return int(match[1]) * 1024 if match else None
    except OSError:
        return None


class Client:
    def __init__(self, base, key, model):
        self.base, self.model = base.rstrip("/"), model
        self.headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}

    def request(self, path, payload=None, timeout=60):
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        return urllib.request.urlopen(urllib.request.Request(self.base + path, body, self.headers), timeout=timeout)

    def json(self, path, payload=None):
        with self.request(path, payload) as response:
            return json.load(response)

    def metrics(self):
        with self.request("/metrics", timeout=10) as response:
            return parse_metrics(response.read().decode())

    def prompt(self, length, salt, sustained=False):
        expected = {key: hashlib.sha256(f"{salt}:{length}:{key}".encode()).hexdigest()[:10]
                    for key in ("alpha", "beta", "gamma")}
        instruction = "Read this reference document. Return only a JSON object containing the exact recorded values of alpha, beta and gamma. Do not summarize the filler.\n"
        ending = "Return the JSON values now. Use exactly these keys: alpha, beta, gamma."
        if sustained:
            instruction = "Read this reference document. First return a JSON object containing the exact recorded values of alpha, beta and gamma.\n"
            ending = ("Start with the JSON object, without a markdown fence, using exactly the keys alpha, beta, gamma. "
                      "After the JSON, write a detailed 2,000-word guide to planning a community garden, "
                      "with numbered sections covering soil, crops, watering and maintenance. Continue in English.")
        unit = "Background reference: the local inference service stores ordinary records.\n"
        repetitions = max(0, (length - 220) // 14)
        padding = 0
        for _ in range(16):
            half = repetitions // 2
            document = (instruction + f"RECORD alpha = {expected['alpha']}\n" + unit * half
                        + f"RECORD beta = {expected['beta']}\n" + unit * (repetitions - half)
                        + " x" * padding + f"\nRECORD gamma = {expected['gamma']}\n"
                        + ending)
            value = self.json("/tokenize", {"model": self.model, "messages": [{"role": "user", "content": document}],
                             "chat_template_kwargs": {"enable_thinking": False}, "add_generation_prompt": True})
            tokens = value["tokens"]
            difference = length - len(tokens)
            if difference == 0:
                return tokens, expected
            if padding + difference < 0:
                repetitions = max(0, repetitions - max(1, (-difference + 10) // 11))
                padding = 0
            else:
                padding += difference
        raise RuntimeError(f"Could not form exact {length}-token prompt")

    def generate(self, tokens, maximum, timeout, minimum_memory):
        started = time.monotonic()
        first = None
        output, generated, usage, finish = [], [], None, None
        stop = threading.Event()
        violation = []
        minimum_seen = available_memory()
        if minimum_seen is not None and minimum_seen < minimum_memory:
            raise RuntimeError("Request refused: host-memory-floor")
        response = self.request("/v1/completions", {
            "model": self.model, "prompt": tokens, "temperature": 0, "max_tokens": maximum,
            "seed": 123, "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True, "add_special_tokens": False,
        }, timeout=timeout)

        def watch():
            nonlocal minimum_seen
            while not stop.wait(1):
                available = available_memory()
                if available is not None:
                    minimum_seen = min(minimum_seen or available, available)
                if time.monotonic() - started > timeout or (available is not None and available < minimum_memory):
                    violation.append("deadline" if time.monotonic() - started > timeout else "host-memory-floor")
                    transport = getattr(getattr(response.fp, "raw", None), "_sock", None)
                    if transport is not None:
                        try:
                            transport.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    response.close()
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            with response:
                for raw in response:
                    if not raw.startswith(b"data: "):
                        continue
                    data = raw[6:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        delta = choice.get("token_ids") or []
                        text = choice.get("text") or ""
                        if first is None and (delta or text):
                            first = time.monotonic()
                        generated.extend(delta)
                        output.append(text)
                        finish = choice.get("finish_reason") or finish
        finally:
            stop.set()
            watcher.join(timeout=2)
        if violation:
            raise RuntimeError("Request cancelled: " + violation[0])
        if not generated or not usage or finish is None:
            raise RuntimeError("Incomplete stream/token-ID evidence")
        if len(generated) != usage.get("completion_tokens"):
            raise RuntimeError("Output token-ID count differs from API usage")
        elapsed = time.monotonic() - started
        return {"text": "".join(output), "tokenIds": generated, "tokenHash": token_hash(generated),
                "finishReason": finish, "usage": usage, "elapsedSeconds": elapsed,
                "ttftSeconds": first - started if first is not None else None,
                "decodeSeconds": time.monotonic() - first if first is not None else None,
                "hostMinimumAvailableBytes": minimum_seen}


def append(path, record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--key-file", type=Path, default=Path("/data/api-access.json"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--lengths", default="8192,32768,65536,131072,262016")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-output", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--memory-floor-gib", type=float, default=2)
    parser.add_argument("--salt", default="sm75-longctx-v1")
    parser.add_argument("--sustained-decode", action="store_true",
                        help="Require the entire output budget; intentional length stop after correct JSON prefix")
    parser.add_argument("--reset-gpu-cache-before-repeat", action="store_true",
                        help="Idle engine only: retain external cache and require measured CPU-to-GPU bytes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, action="append", default=[],
                        help="Passed JSONL baseline; may be supplied for multiple disjoint length sets")
    args = parser.parse_args()
    baselines = read_baselines(args.baseline)
    if args.repeats < 1 or args.max_output < 1 or args.timeout <= 0:
        raise SystemExit("Repeat count, output budget and timeout must be positive")
    lengths = list(map(int, args.lengths.split(",")))
    if args.baseline:
        for length in lengths:
            for repeat in range(1, args.repeats + 1):
                baseline = baselines.get((length, repeat))
                if (not baseline or baseline["salt"] != args.salt or baseline["maxOutputTokens"] != args.max_output
                        or baseline.get("sustainedDecode", False) != args.sustained_decode):
                    raise SystemExit("Baseline is missing a requested case or uses a different contract")
    key_text = args.key_file.read_text(encoding="utf-8").strip()
    key = json.loads(key_text)["key"] if key_text.startswith("{") else key_text
    client = Client(args.base_url, key, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit("Choose a new output path; never overwrite an earlier run")
    with client.request("/health", timeout=10) as response:
        assert response.status == 200
    for length in lengths:
        if length <= 0 or length + args.max_output > 262144:
            raise SystemExit("Input plus output must fit the explicit 262144-token contract")
        tokens, expected = client.prompt(length, args.salt, args.sustained_decode)
        prompt_hash = token_hash(tokens)
        previous_hash = None
        for repeat in range(args.repeats):
            record = {"purpose": "functional-acceptance-not-performance-benchmark", "model": args.model,
                      "inputTokens": length, "maxOutputTokens": args.max_output, "repeat": repeat + 1,
                      "promptTokenHash": prompt_hash, "expected": expected, "salt": args.salt,
                      "sustainedDecode": args.sustained_decode}
            try:
                if repeat and args.reset_gpu_cache_before_repeat:
                    reset = False
                    for _ in range(10):
                        idle = client.metrics()
                        if idle.get("vllm:num_requests_running", 0) or idle.get("vllm:num_requests_waiting", 0):
                            raise RuntimeError("Refusing cache reset while another request is active")
                        reset = client.json("/reset_prefix_cache?reset_running_requests=false&reset_external=false", {}).get("success")
                        if reset:
                            break
                        time.sleep(1)
                    if not reset:
                        raise RuntimeError("GPU prefix cache still has held blocks")
                    record["gpuCacheResetExternalPreserved"] = True
                before = client.metrics()
                result = client.generate(tokens, args.max_output, args.timeout, int(args.memory_floor_gib * 1024**3))
                after = client.metrics()
                if record.get("gpuCacheResetExternalPreserved"):
                    for _ in range(10):
                        if after.get("vllm:kv_offload_load_bytes_total", 0) > before.get("vllm:kv_offload_load_bytes_total", 0):
                            break
                        time.sleep(1)
                        after = client.metrics()
                record.update(result)
                record["metricsBefore"], record["metricsAfter"] = before, after
                record["counterDeltas"] = {name: after[name] - before.get(name, 0) for name in after if name.endswith("_total")}
                assert result["usage"]["prompt_tokens"] == length, "Actual prompt length mismatch"
                if args.sustained_decode:
                    assert result["finishReason"] == "length", "Did not exercise the entire decode budget"
                    assert result["usage"]["completion_tokens"] == args.max_output, "Incomplete sustained decode"
                    actual, _ = json.JSONDecoder().raw_decode(result["text"].strip())
                else:
                    assert result["finishReason"] == "stop", "Did not stop naturally"
                    actual = json.loads(result["text"].strip())
                assert actual == expected, "Three-position retrieval mismatch"
                if previous_hash is not None:
                    assert result["tokenHash"] == previous_hash, "Repeated output token hash mismatch"
                if args.baseline:
                    baseline = baselines[(length, repeat + 1)]
                    assert baseline["promptTokenHash"] == prompt_hash, "Baseline prompt token mismatch"
                    assert baseline["maxOutputTokens"] == args.max_output, "Baseline output budget mismatch"
                    assert baseline["tokenHash"] == result["tokenHash"], "Baseline output token hash mismatch"
                    record["baselineTokenHashMatched"] = True
                if record.get("gpuCacheResetExternalPreserved"):
                    assert record["counterDeltas"].get("vllm:kv_offload_load_bytes_total", 0) > 0, "No measured CPU KV readback"
                previous_hash = result["tokenHash"]
                record["passed"] = True
            except Exception as error:
                record.update(passed=False, error=f"{type(error).__name__}: {error}")
                append(args.output, record)
                print(json.dumps({"inputTokens": length, "repeat": repeat + 1, "passed": False, "error": record["error"]}), flush=True)
                raise
            append(args.output, record)
            print(json.dumps({key: record[key] for key in ("inputTokens", "repeat", "passed", "elapsedSeconds", "ttftSeconds", "tokenHash", "counterDeltas")}), flush=True)


if __name__ == "__main__":
    main()
