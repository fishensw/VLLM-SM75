"""Installation preserves the already validated inference runtime."""
from pathlib import Path
import importlib.util
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "ultra/source/lmcache/install.py"
spec = importlib.util.spec_from_file_location("sm75_lmcache_install", MODULE)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
LOCK = json.loads(MODULE.with_name("install-lock.json").read_text())
BASE = LOCK["dependencyPins"] | LOCK["protectedPackages"]


def test_matching_environment_only_replaces_explicit_binary_wheels():
    plan = installer.install_plan(LOCK, BASE)
    assert plan["missing"] == []
    assert plan["cupyDistributions"] == ["cupy-cuda12x"]
    assert {w["name"] for w in LOCK["wheels"]} == {"lmcache", "cupy-cuda12x"}


def test_missing_dependencies_are_pinned_and_never_resolve_torch_or_cuda():
    installed = {k: v for k, v in BASE.items() if k not in {"aiofiles", "blake3"}}
    plan = installer.install_plan(LOCK, installed)
    assert plan["missing"] == ["aiofiles==25.1.0", "blake3==1.0.9"]
    assert all(not installer.protected(item.split("==")[0]) for item in plan["missing"])


def test_incompatible_existing_dependency_is_rejected_before_changes():
    with pytest.raises(RuntimeError, match="numpy"):
        installer.install_plan(LOCK, BASE | {"numpy": "3.0.0"})
    assert BASE["numpy"] == "2.2.6"


def test_cupy_shared_namespace_conflict_removes_both_distribution_owners():
    plan = installer.install_plan(LOCK, BASE | {"cupy-cuda13x": "14.1.0"})
    assert plan["cupyDistributions"] == ["cupy-cuda12x", "cupy-cuda13x"]


def test_binary_download_hash_mismatch_stops_before_install(monkeypatch, tmp_path):
    import io
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong wheel"))
    with pytest.raises(RuntimeError, match="SHA256"):
        installer.verify_download(LOCK["wheels"][0], tmp_path / "bad.whl")


def test_reviewed_patch_bundle_has_exact_source_and_diff_hashes():
    manifest, data = installer.patch_bundle()
    assert manifest["upstream_pr_head"] == "f180b9ffce7df45ce3037011d95a22db947fefcb"
    assert "open, unmerged" in manifest["upstream_status"]
    assert b"_SubpagedPackedAttentionViewEdit()" in data


def patch_case(tmp_path):
    import hashlib
    original, patched = b"original test module", b"patched test module"
    target = tmp_path / "cache_module.py"
    target.write_bytes(original)
    manifest = {"base_file_sha256": hashlib.sha256(original).hexdigest(),
                "patched_file_sha256": hashlib.sha256(patched).hexdigest()}
    return target, manifest, original, patched


def test_patch_is_atomic_idempotent_and_removes_only_its_bytecode(tmp_path):
    target, manifest, original, patched = patch_case(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    own = cache / "cache_module.cpython-312.pyc"
    other = cache / "unrelated.cpython-312.pyc"
    own.write_bytes(b"old bytecode")
    other.write_bytes(b"keep")
    assert installer.apply_patch(target, manifest, patched) == "patched"
    assert target.read_bytes() == patched
    assert not own.exists()
    assert other.read_bytes() == b"keep"
    assert installer.apply_patch(target, manifest, patched) == "already-patched"
    assert not list(tmp_path.glob(".sm75-packed4d-*"))


def test_unknown_installed_module_is_not_overwritten(tmp_path):
    target, manifest, original, patched = patch_case(tmp_path)
    target.write_bytes(b"local custom modification")
    with pytest.raises(RuntimeError, match="Unknown.*SHA256"):
        installer.apply_patch(target, manifest, patched)
    assert target.read_bytes() == b"local custom modification"


def test_tampered_patch_does_not_change_original_module(tmp_path):
    target, manifest, original, patched = patch_case(tmp_path)
    with pytest.raises(RuntimeError, match="Bundled.*SHA256"):
        installer.apply_patch(target, manifest, b"tampered")
    assert target.read_bytes() == original


def test_atomic_replace_failure_keeps_original_and_cleans_temporary(monkeypatch, tmp_path):
    target, manifest, original, patched = patch_case(tmp_path)
    def fail(*args):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(installer.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        installer.apply_patch(target, manifest, patched)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".sm75-packed4d-*"))


def test_startup_probe_rejects_stock_or_unknown_module_before_cuda_import(monkeypatch, tmp_path):
    probe_path = ROOT / "ultra/source/console/lmcache_probe.py"
    probe_spec = importlib.util.spec_from_file_location("sm75_lmcache_probe", probe_path)
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)
    target = tmp_path / "stock_module.py"
    target.write_bytes(b"not the reviewed packed4D backport")
    class Distribution:
        def locate_file(self, name):
            assert name == "lmcache/integration/vllm/kv_cache_group_edits.py"
            return target
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: {"lmcache": "0.5.5", "vllm": "0.30.0+cu129"}[name])
    monkeypatch.setattr(probe.importlib.metadata, "distribution", lambda name: Distribution())
    with pytest.raises(RuntimeError, match="packed4D"):
        probe.probe({"serverArgs": []})


def old_distribution(tmp_path, *, tracked=True, changed=False):
    import base64
    import hashlib
    from types import SimpleNamespace
    target = tmp_path / "kv_cache_group_edits.py"
    original = b"stock 0.5.4 source"
    digest = base64.urlsafe_b64encode(hashlib.sha256(original).digest()).rstrip(b"=").decode()
    record = installer.metadata.PackagePath("lmcache/integration/vllm/kv_cache_group_edits.py")
    record.hash = SimpleNamespace(mode="sha256", value=digest)
    if tracked:
        target.write_bytes(b"local edit" if changed else original)
    return SimpleNamespace(files=[record] if tracked else [], locate_file=lambda name: target), target


def test_stock_054_upgrade_validates_its_own_record(tmp_path):
    dist, _ = old_distribution(tmp_path)
    manifest, _ = installer.patch_bundle()
    assert installer.check_existing_lmcache({"lmcache": "0.5.4"}, manifest, dist) == "upgrade-stock-0.5.4"


def test_modified_054_module_is_preserved(tmp_path):
    dist, target = old_distribution(tmp_path, changed=True)
    manifest, _ = installer.patch_bundle()
    with pytest.raises(RuntimeError, match="Modified.*0.5.4"):
        installer.check_existing_lmcache({"lmcache": "0.5.4"}, manifest, dist)
    assert target.read_bytes() == b"local edit"


def test_054_without_new_module_can_upgrade_but_untracked_file_is_preserved(tmp_path):
    dist, target = old_distribution(tmp_path, tracked=False)
    manifest, _ = installer.patch_bundle()
    assert installer.check_existing_lmcache({"lmcache": "0.5.4"}, manifest, dist) == "upgrade-stock-0.5.4"
    target.write_bytes(b"custom untracked module")
    with pytest.raises(RuntimeError, match="Untracked"):
        installer.check_existing_lmcache({"lmcache": "0.5.4"}, manifest, dist)


def test_future_or_unrecognized_version_is_not_downgraded(tmp_path):
    dist, _ = old_distribution(tmp_path)
    manifest, _ = installer.patch_bundle()
    with pytest.raises(RuntimeError, match="Only absent"):
        installer.check_existing_lmcache({"lmcache": "0.6.0"}, manifest, dist)


def test_headless_build_still_checks_native_abi(monkeypatch):
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    installer.verify_imports(build_no_gpu=True)
    assert len(calls) == 1
    assert "import lmcache.cuda_ops" in calls[0][0][-1]
    assert calls[0][1]["check"] is True


def test_default_runtime_verification_checks_driver_backed_connector(monkeypatch):
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    installer.verify_imports()
    assert len(calls) == 2
    assert "lmcache_mp_connector" in calls[1][0][-1]
    assert calls[1][1]["check"] is True


def test_headless_build_does_not_suppress_native_import_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise installer.subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(installer.subprocess, "run", fail)
    with pytest.raises(installer.subprocess.CalledProcessError):
        installer.verify_imports(build_no_gpu=True)
