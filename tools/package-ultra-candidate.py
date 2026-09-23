"""Pack the complete candidate build context and a credential-free manifest."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile

root = Path(__file__).resolve().parents[1]
context = root / 'ultra'
files = sorted(p for directory in ['source']
               for p in (context / directory).rglob('*')
               if p.is_file() and not {'data', '__pycache__', 'node_modules', '.pytest_cache', '.ruff_cache'}.intersection(p.parts)
               and not any(part == '.env' or part.startswith('.env.') for part in p.relative_to(context).parts)
               and p.suffix not in {'.log', '.pyc'})
files += [context / name for name in [
    'Dockerfile.candidate', '.dockerignore', 'overlay/font-scale.py',
    'harness/package.json', 'harness/package-lock.json', 'harness/dependencies.json',
]]
manifest = {p.relative_to(context).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
output = Path(sys.argv[1])
manifest_path = output.with_suffix('.manifest.json')
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
with tarfile.open(output, 'w') as archive:
    for source in sorted(files):
        archive.add(source, arcname=source.relative_to(context).as_posix(), recursive=False)
print(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
