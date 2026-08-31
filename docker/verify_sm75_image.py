# SPDX-License-Identifier: Apache-2.0
"""Verify and record the fixed runtime contract for the SM75 image."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch

EXPECTED_PACKAGES = {
    "vllm": "0.28.0",
    "flashinfer-python": "0.6.18",
    "flashinfer-cubin": "0.6.18",
}


def main() -> None:
    installed = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    for name, expected in EXPECTED_PACKAGES.items():
        if installed[name].partition("+")[0] != expected:
            raise RuntimeError(f"Expected {name} {expected}, found {installed[name]}")
    try:
        jit_cache_version = importlib.metadata.version("flashinfer-jit-cache")
    except importlib.metadata.PackageNotFoundError:
        jit_cache_version = None
    if jit_cache_version is not None:
        raise RuntimeError(
            "flashinfer-jit-cache must be absent on SM75; found "
            f"{jit_cache_version}"
        )
    if torch.__version__.partition("+")[0] != "2.13.0":
        raise RuntimeError(f"Expected Torch 2.13.0, found {torch.__version__}")
    if torch.version.cuda != "12.9":
        raise RuntimeError(f"Expected CUDA 12.9, found {torch.version.cuda}")

    result = {
        "packages": installed,
        "flashinfer_jit_cache": "absent",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "target_compute_capability": "7.5",
    }
    output = Path("/opt/vllm-sm75/evidence/runtime-contract.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
