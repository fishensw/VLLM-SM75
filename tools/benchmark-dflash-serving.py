#!/usr/bin/env python3
"""Six serial, fixed long-output requests to an already-running loopback service.
Run this identical file separately for FP8 and its same-weights BF16 control.
Does not launch/toggle engines. Requires an otherwise idle test service.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

PROMPTS = (
    "Write a detailed explanation of how a public library could keep lending books during a week-long computer outage. Use about 600 words of continuous prose. Explain the sequence of decisions, record keeping, staff responsibilities, privacy, common mistakes, and reconciliation after systems recover. Include concrete examples and tradeoffs rather than a brief checklist.",
    "请为一家有三十名员工的小型公司撰写一份约一千二百字的内部说明，解释如何在不增加预算的情况下改善新员工入职体验。请连续展开说明第一天、第一周和第一个月的安排、导师职责、知识记录、远程协作、反馈及效果评估。给出具体情境和取舍，避免只列标题或给出简短总结。",
    "Write a complete Python module using only the standard library that reads an inventory CSV with columns item, category, quantity, and unit_price, validates each row, aggregates quantity and total value by category, and prints a deterministic report. Include type hints, helpful errors with row numbers, Decimal arithmetic, a command-line entry point, and several unittest cases for malformed and valid input. Return the module code directly and implement all parts in full.",
)
COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
LABEL = re.compile(r'(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*")\s*')


def load_smoke(source_root):
    path = source_root / "tools/validate-sm75-serving.py"
    spec = importlib.util.spec_from_file_location("sm75_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def selected_series(sample, model):
    """Select exact metric families/model, retaining engine and position labels."""
    result = {}
    for key, value in sample.get("series", {}).items():
        name = key.split("{", 1)[0]
        if name not in (*COUNTERS.values(), POSITION):
            continue
        labels = {}
        if "{" in key:
            raw = key.split("{", 1)[1].removesuffix("}")
            for match in LABEL.finditer(raw):
                labels[match[1]] = json.loads(match[2])
        if labels.get("model_name") != model:
            continue
        canonical = name + json.dumps(labels, sort_keys=True, separators=(",", ":"))
        result[canonical] = value
    return result


def null_rates():
    return {"token_acceptance_rate": None, "accepted_tokens_per_draft": None,
            "mean_acceptance_length_including_verification_token": None}


def rates(counts):
    return {
        "token_acceptance_rate": counts["accepted_tokens"] / counts["draft_tokens"] if counts["draft_tokens"] else None,
        "accepted_tokens_per_draft": counts["accepted_tokens"] / counts["drafts"] if counts["drafts"] else None,
        "mean_acceptance_length_including_verification_token": 1 + counts["accepted_tokens"] / counts["drafts"] if counts["drafts"] else None,
    }


def metric_delta(before, after, model):
    left, right = selected_series(before, model), selected_series(after, model)
    errors = []
    if before.get("error") or after.get("error"):
        errors.append("Metrics HTTP request failed")
    if set(left) != set(right):
        errors.append("Metric series changed; delta is not comparable")
    delta = {key: right[key] - left[key] for key in left.keys() & right.keys()}
    if any(not math.isfinite(value) or value < 0 for value in delta.values()):
        errors.append("Counter reset or invalid delta")
    counts = {}
    for label, metric in COUNTERS.items():
        values = [value for key, value in delta.items() if key.startswith(metric + "{")]
        if not values:
            errors.append("Missing model counter: " + metric)
        counts[label] = sum(values)
    if counts["accepted_tokens"] > counts["draft_tokens"]:
        errors.append("Accepted tokens exceed drafted tokens")
    positions = [value for key, value in delta.items() if key.startswith(POSITION + "{")]
    if positions and sum(positions) != counts["accepted_tokens"]:
        errors.append("Per-position accepted count does not match total")
    if counts["draft_tokens"] <= 0 or counts["drafts"] <= 0:
        errors.append("No speculative tokens/drafts observed in this request window")
    return {"valid": not errors, "errors": errors, "series": delta, **counts,
            **(rates(counts) if not errors else null_rates())}


def settled_metrics(smoke, client, before, model):
    """Allow asynchronous counter delivery without interpreting absent counts as 0."""
    observations = []
    deadline = time.monotonic() + 3.0
    previous, repeats = None, 0
    while True:
        sample = smoke.metrics(client)
        observations.append(sample)
        current = selected_series(sample, model)
        delta = metric_delta(before, sample, model)
        repeats = repeats + 1 if current == previous else 0
        if sample.get("error") or (delta["valid"] and repeats >= 2) or time.monotonic() >= deadline:
            return sample, observations, delta
        previous = current
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))


def summarize(cases):
    valid = bool(cases) and all(row["metrics_delta"]["valid"] for row in cases)
    counts = {name: sum(row["metrics_delta"][name] for row in cases) for name in COUNTERS}
    seconds = sum(row["elapsed_seconds"] for row in cases)
    tokens = sum((row.get("usage") or {}).get("completion_tokens", 0) for row in cases)
    return {"requests": len(cases), "passed_requests": sum(row["passed"] for row in cases),
            "completion_tokens": tokens, "request_seconds_sum": seconds,
            "completion_tokens_per_request_second": tokens / seconds if seconds else None,
            "outputs_below_128_tokens": sum((row.get("usage") or {}).get("completion_tokens", 0) < 128 for row in cases),
            "outputs_at_token_limit": sum(row.get("finish_reason") == "length" for row in cases),
            "metrics_valid": valid, **counts, **(rates(counts) if valid else null_rates())}


def run(smoke, smoke_path, client, args):
    canonical = [smoke.prompt_payload("MODEL_PLACEHOLDER", prompt, 256, 0) for prompt in PROMPTS]
    report = {"contract": "sm75-long-serving-comparison-v1", "started_at": smoke.utc_now(),
              "label": args.label, "cases": [], "parameters": {
                  "base_url": client.base, "model": args.model, "api_key_env": args.api_key_env,
                  "timeout_seconds": client.timeout, "rounds": 2, "concurrency": 1,
                  "temperature": 0, "seed": 0, "max_tokens": 256, "enable_thinking": False,
                  "metrics_settle_seconds": 3.0, "metrics_poll_seconds": 0.2,
                  "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "smoke_transport_sha256": hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
                  "requests_without_model_sha256": hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode()).hexdigest()},
              "interpretation": [
                  "Token acceptance rate = sum(delta accepted tokens) / sum(delta draft tokens); never average request percentages.",
                  "Mean acceptance length = 1 + delta accepted tokens / delta drafts; the added one is the verification token convention.",
                  "Request throughput includes prefill, generation and HTTP time; it is not decode-only throughput.",
                  "This service must have no other traffic; metrics are model-global, not request-local.",
                  "Round 1 may include cold compilation/cache costs; preserve both rounds separately.",
                  "Length-limited or short output and three synthetic prompts do not establish general chat acceptance or a root cause.",
                  "Matching requests do not prove matching server settings/checkpoint provenance; retain engine configuration and weights evidence separately.",
              ]}

    def checkpoint():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(client.clean(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.output)

    checkpoint()
    try:
        discovery = client.exchange("/v1/models")
        report["model_discovery"] = discovery
        if discovery.get("error"):
            raise RuntimeError(discovery["error"])
        ids = [row.get("id") for row in discovery.get("response", {}).get("data", [])]
        model = args.model
        if model is None:
            if len(ids) != 1 or not isinstance(ids[0], str):
                raise ValueError("Pass --model when the service does not advertise exactly one model")
            model = ids[0]
        if model not in ids:
            raise ValueError("Requested model is not advertised by /v1/models")
        report["parameters"]["model"] = model
        initial = client.exchange("/monitor/spec_decode", monitor=True)
        report["monitor_before"] = initial
        state = initial.get("response") or {}
        if not initial.get("error") and (state.get("enabled") is not True or state.get("pending") or not state.get("spec_configured")):
            raise RuntimeError("Speculative decoding is not configured, enabled and settled; no generation sent")
        for round_number in (1, 2):
            for prompt_number, prompt in enumerate(PROMPTS, 1):
                before = smoke.metrics(client)
                if before.get("error") or not selected_series(before, model):
                    raise RuntimeError("Required model speculative counters unavailable before request")
                record = client.exchange("/v1/chat/completions", smoke.prompt_payload(model, prompt, 256, 0))
                record.update(case=f"round_{round_number}_prompt_{prompt_number}", round=round_number, prompt=prompt_number)
                smoke.check_generation(record)
                after, observations, delta = settled_metrics(smoke, client, before, model)
                record.update(metrics_before=before, metrics_after=after, metrics_observations=observations,
                              metrics_delta=delta, completion_tokens_per_request_second=(record.get("usage") or {}).get("completion_tokens", 0) / record["elapsed_seconds"] if record["elapsed_seconds"] else None)
                report["cases"].append(record)
                checkpoint()
                if not record["passed"] or not delta["valid"]:
                    raise RuntimeError("Request or metrics failed; stopped to preserve the bounded comparison")
        report["monitor_after"] = client.exchange("/monitor/spec_decode", monitor=True)
        end_state = report["monitor_after"].get("response") or {}
        if not report["monitor_after"].get("error") and (end_state.get("enabled") is not True or end_state.get("pending")):
            raise RuntimeError("Speculative state changed during the run")
    except Exception as error:
        report["fatal_error"] = str(error)
    report["finished_at"] = smoke.utc_now()
    report["rounds"] = {str(number): summarize([row for row in report["cases"] if row["round"] == number]) for number in (1, 2)}
    report["summary"] = summarize(report["cases"])
    report["passed"] = not report.get("fatal_error") and len(report["cases"]) == 6 and report["summary"]["passed_requests"] == 6 and report["summary"]["metrics_valid"]
    checkpoint()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8160")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--model")
    parser.add_argument("--label", required=True, help="e.g. fp8-draft or same-weights-bf16-draft")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new evidence file")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env) or not 0 < args.timeout <= 1800:
        parser.error("Invalid API-key environment variable name or timeout")
    smoke, smoke_path = load_smoke(args.source_root)
    try:
        client = smoke.Client(args.base_url, os.environ.get(args.api_key_env, ""), args.timeout)
    except ValueError as error:
        parser.error(str(error))
    report = run(smoke, smoke_path, client, args)
    for number, summary in report["rounds"].items():
        ar = summary["token_acceptance_rate"]
        print(f"round {number}: {summary['completion_tokens']} tokens; {summary['request_seconds_sum']:.3f}s request time; AR={ar:.4%}" if ar is not None else f"round {number}: no valid acceptance rate")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['summary']['passed_requests']}/6 requests; evidence: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
