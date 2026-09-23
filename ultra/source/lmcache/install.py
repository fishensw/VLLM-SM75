"""Opt-in LMCache installation without resolving or replacing the inference stack."""
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def versions():
    return {canonicalize_name(d.metadata["Name"]): d.version for d in metadata.distributions() if d.metadata.get("Name")}


def protected(name):
    name = canonicalize_name(name)
    return name in {"torch", "torchvision", "torchaudio", "vllm"} or name.startswith(("nvidia-", "flashinfer-"))


def install_plan(lock, installed):
    binaries = {w["name"] for w in lock["wheels"]}
    missing = []
    for name, version in lock["dependencyPins"].items():
        name = canonicalize_name(name)
        if protected(name) or name in binaries:
            continue
        if name not in installed:
            missing.append(f"{name}=={version}")
    problems = []
    for entry in lock["resolvedRequirements"] + lock.get("transitiveRequirements", []):
        requirement = Requirement(entry["requirement"])
        name = canonicalize_name(requirement.name)
        if name in binaries:
            continue
        version = installed.get(name, lock["dependencyPins"].get(name))
        if version is None or (requirement.specifier and version not in requirement.specifier):
            problems.append(f"{entry.get('from', 'lmcache')}: {requirement} (installed {version})")
    if problems:
        raise RuntimeError("Existing dependencies are incompatible; no inference packages were changed: " + "; ".join(sorted(set(problems))))
    conflicts = sorted(name for name in installed if name == "cupy" or name.startswith("cupy-cuda") or name.startswith("cupy-rocm"))
    # Different CuPy distributions own the same files; remove both before installing
    # the selected one. An ordinary pip install would leave overlapping ownership.
    return {"missing": sorted(missing), "cupyDistributions": conflicts}


def verify_download(wheel, destination):
    req = urllib.request.Request(wheel["url"], headers={"User-Agent": "vllm-sm75-lmcache-installer"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != wheel["sha256"]:
        raise RuntimeError("Wheel SHA256 mismatch: " + wheel["name"])


def patch_bundle(directory=None):
    directory = Path(directory) if directory else Path(__file__).with_name("patches")
    manifest = json.loads((directory / "packed4d-provenance.json").read_text())
    data = (directory / "kv_cache_group_edits.py").read_bytes()
    if manifest["module_path"] != "lmcache/integration/vllm/kv_cache_group_edits.py":
        raise RuntimeError("Unexpected LMCache patch target")
    if hashlib.sha256(data).hexdigest() != manifest["patched_file_sha256"]:
        raise RuntimeError("Bundled LMCache patch SHA256 mismatch")
    if hashlib.sha256((directory / "packed4d.patch").read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise RuntimeError("Bundled LMCache patch provenance SHA256 mismatch")
    return manifest, data


def verify_patch_target(target, manifest):
    digest = hashlib.sha256(Path(target).read_bytes()).hexdigest()
    if digest not in {manifest["base_file_sha256"], manifest["patched_file_sha256"]}:
        raise RuntimeError("Unknown LMCache module SHA256; refusing to replace local changes")
    return digest


def check_existing_lmcache(installed, manifest, distribution=None):
    version = installed.get("lmcache")
    if version is None:
        return "not-installed"
    dist = distribution or metadata.distribution("lmcache")
    target = Path(dist.locate_file(manifest["module_path"]))
    if version == "0.5.5":
        verify_patch_target(target, manifest)
        return "known-0.5.5"
    if version != "0.5.4":
        raise RuntimeError("Only absent LMCache, stock 0.5.4, or the reviewed 0.5.5 module may be upgraded")
    # The v0.30 base can ship LMCache 0.5.4. Check its own wheel RECORD rather
    # than comparing old source against 0.5.5's hash. Keep untracked/edited files.
    records = dist.files
    if records is None:
        raise RuntimeError("LMCache 0.5.4 has no installation RECORD; refusing to replace unknown files")
    record = next((item for item in records if str(item).replace("\\", "/") == manifest["module_path"]), None)
    if record is None:
        if target.exists():
            raise RuntimeError("Untracked LMCache patch target; refusing to replace local changes")
        return "upgrade-stock-0.5.4"
    if record.hash is None or record.hash.mode != "sha256":
        raise RuntimeError("LMCache 0.5.4 target has no SHA256 RECORD")
    digest = base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest()).rstrip(b"=").decode()
    if digest != record.hash.value:
        raise RuntimeError("Modified LMCache 0.5.4 module; refusing to replace local changes")
    return "upgrade-stock-0.5.4"


def apply_patch(target, manifest, data):
    target = Path(target)
    digest = verify_patch_target(target, manifest)
    if hashlib.sha256(data).hexdigest() != manifest["patched_file_sha256"]:
        raise RuntimeError("Bundled LMCache patch SHA256 mismatch")
    if digest == manifest["patched_file_sha256"]:
        return "already-patched"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".sm75-packed4d-", dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, target.stat().st_mode & 0o777)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["patched_file_sha256"]:
        raise RuntimeError("Installed LMCache patch SHA256 mismatch")
    # A different source size invalidates CPython's ordinary timestamp pyc cache.
    # Remove only the exact module bytecode if present, also covering hash-based pyc.
    for bytecode in target.with_name("__pycache__").glob(target.stem + ".*.pyc"):
        bytecode.unlink()
    return "patched"


def verify_imports(*, build_no_gpu=False):
    # Always load the compiled native libraries: unresolved Torch/CUDA symbols
    # must fail the build. The MP connector also loads Mooncake's driver-backed
    # transport, which requires libcuda.so.1 supplied by the GPU runtime.
    subprocess.run([sys.executable, "-c", "import lmcache.cuda_ops; import lmcache.lmcache_native; print('LMCache native extensions loaded')"], check=True)
    if build_no_gpu:
        print(json.dumps({"nativeImports": "passed", "gpuConnector": "pending-runtime-validation"}))
        return
    subprocess.run([sys.executable, "-c", "import lmcache.integration.vllm.lmcache_mp_connector; print('LMCache external MP connector loaded')"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Apply the checked installation plan; otherwise only inspect")
    parser.add_argument("--build-no-gpu", action="store_true", help="Build-only native checks; the MP connector must pass the GPU runtime probe before use")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("The pinned wheel requires Linux x86_64 and Python 3.12")
    import torch
    if torch.__version__ != "2.13.0+cu129" or torch.version.cuda != "12.9":
        raise RuntimeError("Keep the tested torch 2.13.0+cu129 / CUDA 12.9 stack; this installer never replaces it")
    if metadata.version("vllm").split("+")[0] != "0.30.0":
        raise RuntimeError("This candidate requires vLLM 0.30.0")
    lock = json.loads(Path(__file__).with_name("install-lock.json").read_text())
    patch, patch_data = patch_bundle()
    before = versions()
    prior_module = check_existing_lmcache(before, patch)
    plan = install_plan(lock, before)
    print(json.dumps({"mode": "install" if args.install else "check", "plan": plan, "existingLMCache": prior_module, "wheels": [{k: w[k] for k in ("name", "version", "sha256")} for w in lock["wheels"]]}, indent=2))
    if not args.install:
        return
    with tempfile.TemporaryDirectory(prefix="sm75-lmcache-") as tmp:
        paths = []
        for wheel in lock["wheels"]:
            target = Path(tmp) / wheel["filename"]
            verify_download(wheel, target)
            paths.append(str(target))
        pip = [sys.executable, "-m", "pip"]
        # Download all missing, pinned dependencies before changing installed files.
        if plan["missing"]:
            subprocess.run(pip + ["download", "--no-deps", "--only-binary=:all:", "--dest", tmp, *plan["missing"]], check=True)
        if plan["cupyDistributions"]:
            subprocess.run(pip + ["uninstall", "--yes", *plan["cupyDistributions"]], check=True)
        if plan["missing"]:
            subprocess.run(pip + ["install", "--no-deps", "--no-index", "--find-links", tmp, *plan["missing"]], check=True)
        subprocess.run(pip + ["install", "--no-deps", "--no-index", "--force-reinstall", *paths], check=True)
    after = versions()
    if {k: v for k, v in before.items() if protected(k)} != {k: v for k, v in after.items() if protected(k)}:
        raise RuntimeError("Protected inference package versions changed unexpectedly")
    install_plan(lock, after)
    target = metadata.distribution("lmcache").locate_file(patch["module_path"])
    patch_result = apply_patch(target, patch, patch_data)
    print(json.dumps({"patch": patch["patch_id"], "result": patch_result, "sha256": patch["patched_file_sha256"], "upstreamStatus": patch["upstream_status"]}))
    verify_imports(build_no_gpu=args.build_no_gpu)
    print("Installed LMCache 0.5.5 cu129 + reviewed packed4D backport; inference package versions preserved")


if __name__ == "__main__":
    main()
