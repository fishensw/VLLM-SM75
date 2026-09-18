"""Read-only source/runtime inventory; no torch import, GPU access or credentials.

Build expectations locally, then pipe this script to an image's python3 with
`inspect`. Compare the returned JSON locally. Hashes refer to exact file bytes.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected(repo):
    tree = ast.parse((repo / "docker/helpers/install_sm75_overlay.py").read_text(encoding="utf-8"))
    files = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "files" for t in n.targets))
    records = {}

    def add(source, target, order, purpose):
        records[target] = dict(source=source.relative_to(repo).as_posix(),
                               target=target, sha256=digest(source),
                               order=order, purpose=purpose)

    for relative in files:
        add(repo / "vllm" / relative, "vllm/" + relative, 1, "sm75-overlay")
    for source in sorted((repo / "vllm/third_party/flash_qla_sm75").rglob("*")):
        if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
            add(source, source.relative_to(repo).as_posix(), 1, "flashqla-vendored-source")
    for source in sorted((repo / "docker/speculative").rglob("*.py")):
        add(source, source.relative_to(repo / "docker/speculative").as_posix(), 2, "speculative-overlay")
    standard = dict(records)
    add(repo / "ultra/overlay/monitor.py", "vllm/entrypoints/serve/instrumentator/monitor.py", 3, "ultra-monitor-override")
    return {"sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "standard": list(standard.values()), "ultra": list(records.values())}


def inspect_runtime():
    # All examined paths are source/library files; never read runtime /data.
    roots = [Path(p) for p in sys.path if p and Path(p, "vllm/__init__.py").is_file()]
    if len(set(roots)) != 1:
        raise RuntimeError(f"Expected one installed vllm package, found {len(set(roots))}")
    site = roots[0]
    hashes = {}
    for package in ("vllm", "flashinfer"):
        for path in sorted((site / package).rglob("*")):
            if path.is_file() and (path.name == "LICENSE" or path.suffix in (".py", ".cu", ".cuh", ".h", ".cpp", ".html", ".md", ".txt")):
                hashes[path.relative_to(site).as_posix()] = digest(path)
    for path in sorted(site.glob("sm75_*.py")):
        hashes[path.name] = digest(path)
    for path in sorted(site.glob("flash_qla_sm75*.so")):
        hashes[path.name] = digest(path)
    hooks = {}
    for relative, marker in {
        "vllm/envs.py": "vllm.envs_sm75.apply()",
        "vllm/device_allocator/sleep_mode_backend.py": '"vllm.device_allocator.disk_sleep"',
    }.items():
        file = site / relative
        hooks[relative] = file.is_file() and marker in file.read_text(encoding="utf-8")
    return {"packageRoot": str(site), "sha256": hashes, "hooks": hooks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    gen = sub.add_parser("expected")
    gen.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    sub.add_parser("inspect")
    compare = sub.add_parser("compare")
    compare.add_argument("expected", type=Path)
    compare.add_argument("actual", type=Path)
    compare.add_argument("--variant", choices=("standard", "ultra"), required=True)
    compare.add_argument("--image-id", required=True)
    compare.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "expected":
        print(json.dumps(expected(args.repo), indent=2))
    elif args.action == "inspect":
        print(json.dumps(inspect_runtime(), indent=2))
    else:
        exp = json.loads(args.expected.read_text(encoding="utf-8-sig"))
        actual = json.loads(args.actual.read_text(encoding="utf-8-sig"))
        rows = []
        for record in exp[args.variant]:
            observed = actual["sha256"].get(record["target"])
            rows.append(dict(sourceCommit=exp["sourceCommit"], variant=args.variant,
                             imageId=args.image_id, **record, actualSha256=observed,
                             status="match" if observed == record["sha256"] else "missing" if observed is None else "different"))
        with args.csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({"files": len(rows), "match": sum(r["status"] == "match" for r in rows),
                          "differences": [r["target"] for r in rows if r["status"] != "match"],
                          "hooks": actual["hooks"]}, indent=2))
        if any(r["status"] != "match" for r in rows) or not all(actual["hooks"].values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
