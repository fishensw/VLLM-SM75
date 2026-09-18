"""Check the publishable source tree, not GPU performance or release approval."""
from pathlib import Path
import json
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    errors = []
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    files = [name for name in files if name and (root / name).is_file()]
    for name in files:
        if name.startswith(('evidence/', 'ultra/source/console/data/', '.env')):
            errors.append('Private runtime artifact tracked: ' + name)
        if re.search(r'^docker/(?:Dockerfile.*v0\.|BUILD-v)', name):
            errors.append('Versioned build recipe: ' + name)
        if not name.endswith('.md'):
            continue
        text = (root / name).read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            target = target.strip().split(' "')[0].strip('<>')
            if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith(('#', '/')):
                continue
            path = unquote(target.split('#')[0].split('?')[0])
            if path and not ((root / name).parent / path).exists():
                errors.append(f'{name}: missing link {target}')
    version = (root / 'docker/VERSION').read_text().strip()
    manifest = json.loads((root / f'docs/releases/v{version}-release-manifest.json').read_text())
    if version != manifest['release']:
        errors.append('VERSION and release manifest disagree')
    for name in ['README.md', 'README.en.md', 'docker/BUILD.md', 'ultra/README.md', f'docs/releases/v{version}.md']:
        if not (root / name).is_file():
            errors.append('Missing current guide: ' + name)
    return {'version': version, 'trackedFilesChecked': len(files), 'errors': errors,
            'scope': 'source packaging and links only; performance/release gates remain separate'}


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result['errors']))
