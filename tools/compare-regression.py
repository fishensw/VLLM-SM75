"""Compare complete paired benchmark contracts; fail closed on missing evidence."""
import argparse
import json
import statistics
from pathlib import Path


def compare(groups, tolerance=0.05, repeats=5):
    errors, summaries = [], {}
    baseline = next(iter(groups))
    shapes = sorted({r['inputTokens'] for r in groups[baseline]})
    if not shapes:
        errors.append('Empty baseline')
    for label, rows in groups.items():
        summaries[label] = {}
        if sorted({r['inputTokens'] for r in rows}) != shapes:
            errors.append(f'{label}: shape mismatch')
        for shape in shapes:
            samples = [r for r in rows if r['inputTokens'] == shape]
            reference = [r for r in groups[baseline] if r['inputTokens'] == shape]
            if sorted(r['repeat'] for r in samples) != list(range(1, repeats + 1)):
                errors.append(f'{label}/{shape}: incomplete or duplicate repeats')
            if not samples:
                continue
            if any(not r['valid'] or not r['complete'] or not r['retrievalPassed']
                   or r['cachedTokens'] != 0 or r['preemptions'] != 0 for r in samples):
                errors.append(f'{label}/{shape}: invalid samples')
            contract = lambda r: (r['contract'], r['model'], r['outputTokens'], r['promptTokenHash'])
            if len({contract(r) for r in samples + reference}) != 1:
                errors.append(f'{label}/{shape}: different test contract')
            hash_match = len({r['tokenHash'] for r in samples + reference}) == 1
            if not hash_match:
                errors.append(f'{label}/{shape}: output token mismatch')
            metrics = {}
            for key in ('prefillTokensPerSecond', 'decodeTokensPerSecond', 'ttftSeconds', 'tpotSeconds'):
                median = statistics.median(r[key] for r in samples)
                base = statistics.median(r[key] for r in reference)
                delta = median / base - 1
                metrics[key] = {'median': median, 'baselineChangePercent': delta * 100}
                if key.endswith('TokensPerSecond') and delta < -tolerance:
                    errors.append(f'{label}/{shape}: {key} regressed {delta:.2%}')
            summaries[label][str(shape)] = {'samples': len(samples), 'tokenHashMatches': hash_match, 'metrics': metrics}
    return {'baseline': baseline, 'throughputTolerancePercent': tolerance * 100,
            'passed': not errors, 'errors': errors, 'results': summaries,
            'scope': 'Only these samples; caller must verify identical hardware, runtime arguments and clock/cache conditions.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path, help='Baseline first, then candidates')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.files) < 2 or len({p.stem for p in args.files}) != len(args.files):
        parser.error('Supply a baseline and candidates with distinct file names')
    result = compare({p.stem: [json.loads(line) for line in p.read_text().splitlines() if line.strip()] for p in args.files})
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(not result['passed'])
