#!/usr/bin/env python3
"""Bounded functional smoke checks for an already-running loopback vLLM server.

Does not start an engine or select hardware. API credentials are read only from
an environment variable. Greedy differences are observations, not diagnoses or
an automatic numerical-equivalence gate. Output is checkpointed JSON.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

PROMPTS = (
    "Reply with only the result of 17 * 23.",
    "Translate 'The library is open today.' into Chinese. Reply with the translation only.",
    "Write one Python statement that creates a list containing the integers 1, 2 and 3. No markdown.",
)
SCHEMA = {"type": "object", "properties": {"status": {"type": "string"},
          "count": {"type": "integer"}}, "required": ["status", "count"],
          "additionalProperties": False}
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def loopback_base(value):
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in ("http", "https") or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Use a loopback HTTP(S) URL without credentials, query or fragment")
    host = parts.hostname or ""
    try:
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if not local or parts.path.rstrip("/") not in ("", "/v1"):
        raise ValueError("Base URL must be loopback, optionally ending in /v1")
    # Accessing port also validates malformed/out-of-range port strings.
    _ = parts.port
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "", "", ""))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward an API key outside the explicitly selected service.
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


class Client:
    def __init__(self, base, key="", timeout=120):
        self.base, self.key, self.timeout = loopback_base(base), key, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def clean(self, value):
        if isinstance(value, str):
            return value.replace(self.key, "[REDACTED]") if self.key else value
        if isinstance(value, dict):
            return {self.clean(k): self.clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        return value

    def exchange(self, path, payload=None, stream=False, monitor=False, timeout=None):
        record = {"method": "POST" if payload is not None else "GET", "path": path,
                  "request": payload, "started_at": utc_now(), "http_status": None}
        started = time.monotonic()
        timeout = self.timeout if timeout is None else min(self.timeout, timeout)
        response, timer = None, None
        try:
            headers = {"Content-Type": "application/json"}
            if self.key:
                headers["Authorization"] = "Bearer " + self.key
                if monitor:
                    headers["x-api-key-hash"] = hashlib.sha256(self.key.encode()).hexdigest()
            body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
            request = urllib.request.Request(self.base + path, body, headers)
            try:
                response = self.opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as error:
                response = error
            record["http_status"] = response.code
            expired = threading.Event()

            def close_at_deadline():
                expired.set()
                transport = getattr(getattr(response.fp, "raw", None), "_sock", None)
                if transport:
                    try:
                        transport.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                response.close()

            timer = threading.Timer(max(0.01, timeout - (time.monotonic() - started)), close_at_deadline)
            timer.daemon = True
            timer.start()
            if stream and 200 <= response.code < 300:
                record.update(self.read_stream(response))
            else:
                data = response.read(MAX_RESPONSE_BYTES + 1)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise ValueError("Response exceeds 2 MiB limit")
                text = data.decode("utf-8")
                if path == "/metrics":
                    record["text"] = text
                else:
                    record["response"] = json.loads(text)
            if expired.is_set():
                raise TimeoutError("Request wall-clock deadline exceeded")
            if not 200 <= response.code < 300:
                record["error"] = f"HTTP {response.code}"
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            if timer:
                timer.cancel()
            if response:
                response.close()
            record["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return self.clean(record)

    @staticmethod
    def read_stream(response):
        text, tokens, usage, finish = [], [], None, None
        token_ids_seen, done, count, size = False, False, 0, 0
        for raw in iter(lambda: response.readline(MAX_RESPONSE_BYTES + 1), b""):
            size += len(raw)
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("Stream exceeds 2 MiB limit")
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            count += 1
            if event.get("error"):
                raise ValueError(f"Stream error: {event['error']}")
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                fragment = delta.get("content") or ""
                if not isinstance(fragment, str):
                    raise ValueError("Stream content must be text")
                text.append(fragment)
                ids = choice.get("token_ids", delta.get("token_ids"))
                if ids is not None:
                    token_ids_seen = True
                    tokens.extend(ids)
                finish = choice.get("finish_reason") or finish
        return {"text": "".join(text), "token_ids": tokens if token_ids_seen else None,
                "usage": usage, "finish_reason": finish, "stream_done": done, "stream_events": count}


def prompt_payload(model, prompt, maximum, seed, stream=False):
    result = {"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "top_p": 1, "seed": seed, "max_tokens": maximum,
              "n": 1, "chat_template_kwargs": {"enable_thinking": False},
              "return_token_ids": True, "stream": stream}
    if stream:
        result["stream_options"] = {"include_usage": True}
    return result


def check_generation(record, structured=False):
    errors = []
    if record.get("error"):
        errors.append(record["error"])
    if not record.get("request", {}).get("stream") and not errors:
        data = record.get("response") or {}
        choices = data.get("choices") or []
        if len(choices) != 1:
            errors.append("Expected exactly one completion choice")
        else:
            choice = choices[0]
            record.update(text=(choice.get("message") or {}).get("content"),
                          token_ids=choice.get("token_ids"), usage=data.get("usage"),
                          finish_reason=choice.get("finish_reason"))
    text, usage = record.get("text"), record.get("usage")
    if not isinstance(text, str) or not text.strip():
        errors.append("Completion text is empty or absent")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < minimum
            for k, minimum in (("prompt_tokens", 1), ("completion_tokens", 1), ("total_tokens", 2))):
        errors.append("Missing or invalid token usage")
    elif usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        errors.append("Token usage totals are inconsistent")
    if not record.get("finish_reason"):
        errors.append("Missing finish reason")
    if record.get("request", {}).get("stream") and not record.get("stream_done"):
        errors.append("Stream ended without [DONE]")
    ids = record.get("token_ids")
    if ids is not None:
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            errors.append("Invalid generated token IDs")
        elif isinstance(usage, dict) and len(ids) != usage.get("completion_tokens"):
            errors.append("Generated token-ID count differs from completion usage")
    if structured and isinstance(text, str):
        try:
            obj = json.loads(text)
            if (not isinstance(obj, dict) or set(obj) != {"status", "count"}
                    or not isinstance(obj["status"], str) or type(obj["count"]) is not int):
                raise ValueError("Response does not match requested JSON schema")
            record["structured_result"] = obj
        except (ValueError, TypeError) as error:
            errors.append(str(error))
    record["passed"], record["checks"] = not errors, errors
    return record


def metrics(client):
    record = client.exchange("/metrics")
    series, aggregate = {}, {}
    for line in record.pop("text", "").splitlines():
        match = re.match(r'([^\s{]+)(\{.*\})?\s+(\S+)', line)
        if not match or "spec_decode" not in match[1] or match[1].endswith("_created"):
            continue
        try:
            value = float(match[3])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        series[match[1] + (match[2] or "")] = value
        aggregate[match[1]] = aggregate.get(match[1], 0) + value
    record.update(series=series, aggregate=aggregate)
    return record


def compare(left, right):
    a, b = left.get("token_ids"), right.get("token_ids")
    available = isinstance(a, list) and isinstance(b, list)
    first = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]),
                 min(len(a), len(b)) if a != b else None) if available else None
    return {"case": left["case"], "text_equal": left.get("text") == right.get("text"),
            "generated_token_ids_available": available, "token_ids_equal": a == b if available else None,
            "first_different_token_index": first,
            "completion_tokens_equal": (left.get("usage") or {}).get("completion_tokens")
                                       == (right.get("usage") or {}).get("completion_tokens"),
            "enabled_text": left.get("text"), "disabled_text": right.get("text"),
            "enabled_token_ids": a, "disabled_token_ids": b}


def settled_state(client, enabled, seconds):
    deadline = time.monotonic() + seconds
    while True:
        sample = client.exchange("/monitor/spec_decode", monitor=True, timeout=max(0.01, deadline - time.monotonic()))
        state = sample.get("response") or {}
        if sample.get("error"):
            raise RuntimeError(sample["error"])
        if state.get("enabled") is enabled and not state.get("pending", False):
            return sample
        if time.monotonic() >= deadline:
            raise TimeoutError("Speculative toggle did not settle")
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))


def run(client, model, output, maximum=64, seed=0, toggle_timeout=15, key_env="VLLM_API_KEY"):
    report = {"contract": "sm75-serving-smoke-v1", "started_at": utc_now(),
              "parameters": {"base_url": client.base, "model": model, "api_key_env": key_env,
                             "request_timeout_seconds": client.timeout, "max_tokens": maximum,
                             "seed": seed, "concurrency": 4, "toggle_timeout_seconds": toggle_timeout},
              "cases": [], "metrics": {}, "speculative_comparison": {"status": "not_run"}}

    def checkpoint():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(client.clean(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)

    def case(name, payload, structured=False):
        result = client.exchange("/v1/chat/completions", payload, stream=payload["stream"])
        result["case"] = name
        return check_generation(result, structured)

    checkpoint()
    try:
        discovery = client.exchange("/v1/models")
        report["model_discovery"] = discovery
        if discovery.get("error"):
            raise RuntimeError(discovery["error"])
        ids = [row.get("id") for row in (discovery.get("response") or {}).get("data", []) if isinstance(row, dict)]
        if model is None:
            if len(ids) != 1 or not isinstance(ids[0], str):
                raise ValueError("Pass --model when /v1/models does not expose exactly one model")
            model = ids[0]
        elif model not in ids:
            raise ValueError("Requested --model is not advertised by /v1/models")
        report["parameters"]["model"] = model
        initial = client.exchange("/monitor/spec_decode", monitor=True)
        report["speculative_comparison"]["initial"] = initial
        report["metrics"]["before"] = metrics(client)
        baseline = []
        for i, prompt in enumerate(PROMPTS):
            record = case(f"short_{i + 1}", prompt_payload(model, prompt, maximum, seed))
            baseline.append(record)
            report["cases"].append(record)
            checkpoint()
        report["cases"].append(case("stream", prompt_payload(model, PROMPTS[0], maximum, seed, True)))
        payload = prompt_payload(model, 'Return a JSON object with status "ready" and count 3.', maximum, seed)
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "sm75_smoke", "strict": True, "schema": SCHEMA}}
        report["cases"].append(case("json_schema", payload, True))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(case, f"concurrent_{i + 1}", prompt_payload(model, PROMPTS[i % 3], maximum, seed)) for i in range(4)]
            report["cases"].extend(future.result() for future in futures)
        report["metrics"]["after_enabled"] = metrics(client)
        checkpoint()
        comparison = report["speculative_comparison"]
        state = initial.get("response") or {}
        if initial.get("error") or not state.get("spec_configured") or state.get("enabled") is not True or state.get("pending"):
            comparison.update(status="skipped", reason="Monitor unavailable, speculative decoding unconfigured/disabled, or a toggle is already pending")
        else:
            original = state.get("desired_enabled", state["enabled"])
            changed = False
            try:
                # Mark before POST so a response timeout still triggers restoration.
                changed = True
                comparison["disable"] = client.exchange("/monitor/spec_decode", {"enabled": False}, monitor=True)
                if comparison["disable"].get("error"):
                    raise RuntimeError(comparison["disable"]["error"])
                comparison["disabled_state"] = settled_state(client, False, toggle_timeout)
                controls = [case(f"target_only_{i + 1}", prompt_payload(model, prompt, maximum, seed)) for i, prompt in enumerate(PROMPTS)]
                report["cases"].extend(controls)
                comparison.update(status="completed", results=[compare(a, b) for a, b in zip(baseline, controls)],
                                  note="Text/token differences alone do not identify a root cause; no numerical-equivalence claim is made.")
                report["metrics"]["after_disabled"] = metrics(client)
            except Exception as error:
                comparison.update(status="failed", error=str(error))
            finally:
                if changed:
                    comparison["restore"] = client.exchange("/monitor/spec_decode", {"enabled": original}, monitor=True)
                    try:
                        if comparison["restore"].get("error"):
                            raise RuntimeError(comparison["restore"]["error"])
                        comparison["restored_state"] = settled_state(client, original, toggle_timeout)
                    except Exception as error:
                        comparison["restore_error"] = str(error)
                        comparison["status"] = "failed"
                checkpoint()
    except Exception as error:
        report["fatal_error"] = str(error)
    report["finished_at"] = utc_now()
    report["passed"] = (not report.get("fatal_error") and bool(report["cases"])
                        and all(row["passed"] for row in report["cases"])
                        and report["speculative_comparison"]["status"] != "failed")
    report["summary"] = {"passed_cases": sum(row["passed"] for row in report["cases"]),
                         "total_cases": len(report["cases"]),
                         "greedy_differences": sum(not row["text_equal"] or row["token_ids_equal"] is False
                                                  for row in report["speculative_comparison"].get("results", []))}
    checkpoint()
    return client.clean(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY", help="Environment variable containing the key; no key is written to evidence")
    parser.add_argument("--model", help="Defaults to the sole model advertised by /v1/models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--toggle-timeout", type=float, default=15)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new evidence file")
    if not 1 <= args.max_tokens <= 512 or not 0 < args.timeout <= 1800 or not 0 < args.toggle_timeout <= 120:
        parser.error("max-tokens must be 1..512; timeout 0..1800; toggle-timeout 0..120")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env):
        parser.error("api-key-env must be an environment variable name")
    try:
        client = Client(args.base_url, os.environ.get(args.api_key_env, ""), args.timeout)
    except ValueError as error:
        parser.error(str(error))
    report = run(client, args.model, args.output, args.max_tokens, args.seed, args.toggle_timeout, args.api_key_env)
    summary = report["summary"]
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {summary['passed_cases']}/{summary['total_cases']} cases; "
          f"spec comparison {report['speculative_comparison']['status']}; observed greedy differences {summary['greedy_differences']}")
    print(f"Evidence: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())