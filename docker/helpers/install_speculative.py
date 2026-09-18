"""Install the audited speculative overlay without host cache dependencies."""
import compileall
import hashlib
import json
import shutil
from pathlib import Path

def install(source, package, evidence):
    records = {}
    for src in sorted(source.rglob('*.py')):
        rel = src.relative_to(source)
        dest = package / Path(*rel.parts[1:]) if rel.parts[0] == 'vllm' else package.parent / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        records[rel.as_posix()] = hashlib.sha256(dest.read_bytes()).hexdigest()
        if not compileall.compile_file(str(dest), quiet=1):
            raise RuntimeError(f'Syntax check failed: {rel}')
    if len(records) != 8:
        raise RuntimeError('Expected eight reviewed speculative source files')
    runner = (package / 'v1/worker/gpu/model_runner.py').read_text(encoding='utf-8')
    for hook in ('_sm75_fa2_graph.install()', '_sm75_gdn_meta.install()'):
        if hook not in runner:
            raise RuntimeError(f'Missing activation hook: {hook}')
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


if __name__ == '__main__':
    import vllm
    install(Path('/opt/vllm-sm75/speculative'), Path(vllm.__file__).resolve().parent,
            Path('/opt/vllm-sm75/evidence/speculative-files.json'))
