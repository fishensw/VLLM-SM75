"""Package tracked sources only, plus a combined standard/ultra build graph.

The combined graph makes the ultra FROM reference the exact standard stage;
it does not need to import an intermediate image into the production daemon.
"""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import tarfile

root = Path(__file__).resolve().parents[1]
if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root):
    raise SystemExit("Commit tracked changes before packaging")
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
archive = subprocess.check_output(["git", "archive", "--format=tar", revision], cwd=root)
standard = subprocess.check_output(["git", "show", f"{revision}:docker/Dockerfile"], cwd=root).decode()
ultra = subprocess.check_output(["git", "show", f"{revision}:ultra/Dockerfile"], cwd=root).decode()
node_arg = next(line for line in ultra.splitlines() if line.startswith("ARG NODE_IMAGE="))
combined = standard.replace("ARG BASE_IMAGE=", node_arg + "\nARG BASE_IMAGE=", 1).rstrip() + "\n\n"
for line in ultra.splitlines():
    if line.startswith(("ARG VLLM_IMAGE=", "ARG NODE_IMAGE=")):
        continue
    if line == "FROM ${VLLM_IMAGE}":
        line = "FROM final AS ultra-final"
    if line.startswith("COPY ") and not line.startswith("COPY --from="):
        parts = line.split()
        line = " ".join(["COPY", *["ultra/" + source for source in parts[1:-1]], parts[-1]])
    combined += line + "\n"
assert "FROM final AS ultra-final" in combined
assert "COPY ultra/source /opt/sm75-workbench" in combined
combined += """
FROM ultra-final AS ultra-tests
RUN node --test /opt/sm75-workbench/console/test/*.test.mjs > /tmp/ultra-node-tests.log 2>&1 || { cat /tmp/ultra-node-tests.log; exit 1; }
FROM ultra-final AS ultra-verified
COPY --from=ultra-tests /tmp/ultra-node-tests.log /opt/vllm-sm75/evidence/ultra-node-tests.log
"""
output = Path(sys.argv[1])
with tarfile.open(fileobj=io.BytesIO(archive)) as source, tarfile.open(output, "w") as dest:
    for entry in source:
        if entry.name.startswith(("evidence/", ".git/")):
            raise SystemExit("Private data in tracked build context")
        dest.addfile(entry, source.extractfile(entry) if entry.isfile() else None)
    data = combined.encode()
    entry = tarfile.TarInfo("Dockerfile.full")
    entry.size = len(data)
    entry.mode = 0o644
    dest.addfile(entry, io.BytesIO(data))
manifest = {"sourceRevision": revision, "archiveSha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "combinedDockerfileSha256": hashlib.sha256(data).hexdigest()}
output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest))
