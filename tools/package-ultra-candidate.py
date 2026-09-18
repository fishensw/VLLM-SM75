"""Pack candidate console sources and a credential-free SHA256 manifest."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile

root = Path(__file__).resolve().parents[1]
context = root / 'ultra'
files = sorted(p for directory in ['source/console', 'overlay/dsh-layout']
               for p in (context / directory).rglob('*')
               if p.is_file() and not {'data', '__pycache__', 'node_modules'}.intersection(p.parts)
               and p.suffix not in {'.log', '.pyc'})
manifest = {p.relative_to(context).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
output = Path(sys.argv[1])
manifest_path = output.with_suffix('.manifest.json')
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
with tarfile.open(output, 'w') as archive:
    for source in [context / 'Dockerfile.candidate', context / '.dockerignore', *files]:
        archive.add(source, arcname=source.relative_to(context).as_posix(), recursive=False)
print(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
