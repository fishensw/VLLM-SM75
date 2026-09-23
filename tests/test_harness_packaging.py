"""Candidate archives must satisfy Docker COPY and preserve native UI bundles."""
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_archive_contains_every_local_docker_copy(tmp_path):
    archive = tmp_path / "candidate.tar"
    subprocess.run([sys.executable, str(ROOT / "tools/package-ultra-candidate.py"), str(archive)], check=True)
    with tarfile.open(archive) as packed:
        names = packed.getnames()
        dockerfile = packed.extractfile("Dockerfile.candidate").read().decode()
        manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
        assert set(manifest) == set(names)
        assert "source/vllm/entrypoints/serve/instrumentator/dashboard.html" in names
        for name, digest in manifest.items():
            assert hashlib.sha256(packed.extractfile(name).read()).hexdigest() == digest
        for line in dockerfile.splitlines():
            if not line.startswith("COPY ") or "--from=" in line:
                continue
            for source in line.split()[1:-1]:
                assert any(name == source or name.startswith(source.rstrip("/") + "/") for name in names), source
        assert not any("overlay/dsh-" in name for name in names)
        assert not any(set(Path(name).parts) & {"data", "node_modules", "__pycache__"} for name in names)


def test_font_scaling_handles_mixed_rules_and_is_idempotent(tmp_path):
    bundle = tmp_path / "ui-test/lib/client.js"
    bundle.parent.mkdir(parents=True)
    bundle.write_text('a{font-size:calc(12px * var(--dsh-font-scale,1));font-size:14px;line-height:20px;font:13px/1.2 sans-serif}\n', encoding="utf-8")
    command = [sys.executable, str(ROOT / "ultra/overlay/font-scale.py"), "--base", str(tmp_path)]
    subprocess.run(command, check=True)
    first = bundle.read_bytes()
    assert b"font-size:14px" not in first
    assert first.count(b"var(--dsh-font-scale,1)") == 4
    subprocess.run(command, check=True)
    assert bundle.read_bytes() == first


def test_candidate_archive_excludes_local_environment_and_cache_files(tmp_path):
    source = tmp_path / "source-tree"
    shutil.copytree(ROOT / "ultra", source / "ultra",
                    ignore=shutil.ignore_patterns("node_modules", "data", "__pycache__", ".pytest_cache", ".ruff_cache"))
    (source / "tools").mkdir()
    packer = source / "tools/package-ultra-candidate.py"
    shutil.copyfile(ROOT / "tools/package-ultra-candidate.py", packer)
    private_names = [".env", ".env.local", "nested/.env/secret.txt",
                     ".pytest_cache/results", ".ruff_cache/results"]
    console = source / "ultra/source/console"
    for name in private_names:
        file = console / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("synthetic-private-build-fixture", encoding="utf-8")
    public = console / "public/release-fixture.js"
    public.write_text("export const releaseFixture = true;", encoding="utf-8")
    archive = tmp_path / "candidate.tar"
    subprocess.run([sys.executable, str(packer), str(archive)], check=True)
    with tarfile.open(archive) as packed:
        names = set(packed.getnames())
        assert "source/console/public/release-fixture.js" in names
        for name in private_names:
            assert "source/console/" + name not in names
        assert not any(b"synthetic-private-build-fixture" in packed.extractfile(member).read()
                       for member in packed if member.isfile())
