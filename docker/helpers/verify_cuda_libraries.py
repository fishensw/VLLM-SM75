# SPDX-License-Identifier: Apache-2.0
"""Reject CUDA-family collisions even when installed package metadata looks right."""
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

PACKAGES = {
    "nvidia-cudnn-cu12": "9.20.0.48",
    "nvidia-cusparselt-cu12": "0.8.1",
    "nvidia-nccl-cu12": "2.30.7",
    "nvidia-nvshmem-cu12": "3.4.5",
}


def verify_distribution_libraries(distribution):
    libraries = []
    for record in distribution.files or []:
        if ".so" not in str(record):
            continue
        if record.hash is None or record.hash.mode != "sha256":
            raise RuntimeError(f"Missing SHA256 RECORD for {record}")
        path = distribution.locate_file(record)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
        if encoded != record.hash.value:
            raise RuntimeError(f"CUDA library differs from installed wheel RECORD: {record}")
        libraries.append({"file": str(record), "sha256": digest.hexdigest()})
    if not libraries:
        raise RuntimeError("CUDA package has no recorded shared libraries")
    return libraries


def main():
    evidence = {}
    for name, expected in PACKAGES.items():
        collision = name.replace("-cu12", "-cu13")
        try:
            metadata.distribution(collision)
        except metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError(f"Conflicting CUDA-family package remains: {collision}")
        package = metadata.distribution(name)
        if package.version != expected:
            raise RuntimeError(f"Expected {name} {expected}, found {package.version}")
        evidence[name] = {"version": package.version, "libraries": verify_distribution_libraries(package)}
    for name in ("cuda-python", "cuda-bindings"):
        if metadata.version(name) != "12.9.7":
            raise RuntimeError(f"{name} does not match the CUDA 12.9 runtime")
    import torch
    if torch.__version__ != "2.13.0+cu129" or torch.version.cuda != "12.9":
        raise RuntimeError("Torch CUDA 12.9 ABI was not restored")
    if torch.backends.cudnn.version() != 92000:
        raise RuntimeError("Loaded cuDNN is not the pinned CUDA 12 build")
    output = Path("/opt/vllm-sm75/evidence/cuda-libraries.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print("CUDA 12 shared-library RECORD hashes verified")


if __name__ == "__main__":
    main()
