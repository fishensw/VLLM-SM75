"""Release v2: decode window excludes the first streamed token batch and trailer.

Sequential, exact-token performance samples; credentials are never recorded.

Run against an idle owned engine. Results are measurements, not automatic release
approval: compare identical prompts, settings, hardware and compilation contracts.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import uuid
import time
import hashlib

spec = importlib.util.spec_from_file_location('long_context', Path(__file__).with_name('validate-long-context.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class Client(helpers.Client):
    def request(self, path, payload=None, timeout=60):
        if path == '/v1/completions':
            payload = {**payload, 'ignore_eos': True, 'cache_salt': str(uuid.uuid4())}
        return super().request(path, payload, timeout)


SPEC = {"drafts": "vllm:spec_decode_num_drafts_total", "draftTokens": "vllm:spec_decode_num_draft_tokens_total", "acceptedTokens": "vllm:spec_decode_num_accepted_tokens_total"}
helpers.METRICS.update(SPEC.values())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', default='8192,32768')
    p.add_argument('--require-spec', action='store_true')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output-tokens', type=int, default=512)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; use a new file to avoid mixing experiments')
    if a.repeats < 1 or a.output_tokens < 2:
        p.error('Invalid repeat/output contract')
    try:
        lengths = [int(value) for value in a.lengths.split(',')]
    except ValueError:
        p.error('Lengths must be comma-separated integers')
    if not lengths or min(lengths) < 1 or len(set(lengths)) != len(lengths):
        p.error('Lengths must be positive and unique')
    secret = a.key_file.read_text().strip()
    client = Client(a.base_url, json.loads(secret)['key'] if secret.startswith('{') else secret, a.model)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for length in lengths:
        tokens, expected = client.prompt(length, 'sm75-regression-v1', sustained=True)
        # Shape-specific warmup does not contaminate prefix hit counts: every
        # request has a separate cache salt, including warmup.
        client.generate(tokens, a.output_tokens, 900, 2 * 1024**3)
        time.sleep(1)  # Let warmup counters arrive before the first measured window.
        for repeat in range(1, a.repeats + 1):
            before = client.metrics()
            if before.get('vllm:num_requests_running', 0) or before.get('vllm:num_requests_waiting', 0):
                raise RuntimeError('Engine is not idle')
            result = client.generate(tokens, a.output_tokens, 900, 2 * 1024**3)
            # Read counters after asynchronous metric delivery has settled.
            time.sleep(1)
            after = client.metrics()
            counts = {k: after.get(v, 0) - before.get(v, 0) for k, v in SPEC.items()}
            spec_valid = (all(v >= 0 for v in counts.values()) and
                          counts['drafts'] > 0 and counts['draftTokens'] > 0 and
                          counts['acceptedTokens'] <= counts['draftTokens'])
            acceptance = counts['acceptedTokens'] / counts['draftTokens'] if spec_valid else None
            first_count = result['firstEmissionTokens']
            decode_window = result['decodeWindowSeconds']
            timing_valid = 0 < first_count < a.output_tokens and decode_window > 0
            cached = result['usage'].get('prompt_tokens_details', {}).get('cached_tokens')
            preemptions = after.get('vllm:num_preemptions_total', 0) - before.get('vllm:num_preemptions_total', 0)
            row = {'contract': 'sm75-release-v2', 'label': a.label, 'model': a.model,
                   'benchmarkSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'protocolSha256': hashlib.sha256(Path(helpers.__file__).read_bytes()).hexdigest(),
                   'inputTokens': length, 'outputTokens': a.output_tokens, 'repeat': repeat,
                   'promptTokenHash': helpers.token_hash(tokens), 'tokenHash': result['tokenHash'],
                   'ttftSeconds': result['ttftSeconds'],
                   'prefillTokensPerSecond': length / result['ttftSeconds'],
                   'decodeTokensPerSecond': (a.output_tokens - first_count) / decode_window if timing_valid else None,
                   'tpotSeconds': decode_window / (a.output_tokens - first_count) if timing_valid else None,
                   'firstEmissionTokens': first_count, 'decodeWindowSeconds': decode_window,
                   'speculativeCounters': counts, 'speculativeMetricsValid': spec_valid,
                   'acceptanceRate': acceptance,
                   'elapsedSeconds': result['elapsedSeconds'], 'cachedTokens': cached,
                   'preemptions': preemptions,
                   'retrievalPassed': all(value in result['text'] for value in expected.values()),
                   'complete': len(result['tokenIds']) == a.output_tokens,
                   'hostMinimumAvailableBytes': result['hostMinimumAvailableBytes']}
            row['valid'] = timing_valid and (spec_valid or not a.require_spec) and row['complete'] and cached == 0 and preemptions == 0 and row['retrievalPassed']
            helpers.append(a.output, row)
            print(json.dumps(row), flush=True)
            if not row['valid']:
                raise RuntimeError('Invalid benchmark sample; inspect recorded evidence')


if __name__ == '__main__':
    main()
