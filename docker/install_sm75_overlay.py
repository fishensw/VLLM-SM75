# SPDX-License-Identifier: Apache-2.0
"""Install the reviewed SM75 overlay into the base image's vLLM package."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import vllm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay", type=Path)
    parser.add_argument("--source-copy", type=Path)
    args = parser.parse_args()

    source_root = args.overlay.resolve()
    package_root = Path(vllm.__file__).resolve().parent
    files = [
        "envs.py",
        "config/model.py",
        "engine/arg_utils.py",
        "entrypoints/serve/utils/api_utils.py",
        "distributed/kv_transfer/kv_connector/v1/base.py",
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "model_executor/layers/quantization/kv_cache.py",
        "model_executor/layers/quantization/utils/marlin_utils_fp8.py",
        "v1/attention/backends/flashinfer.py",
        "v1/attention/backends/gdn_attn.py",
        "v1/engine/async_llm.py",
        "v1/engine/auto_sleep.py",
        "v1/engine/core.py",
        "v1/engine/core_client.py",
    ]
    for relative in files:
        source = source_root / relative
        destination = package_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    third_party_source = source_root / "third_party/flash_qla_sm75"
    third_party_destination = package_root / "third_party/flash_qla_sm75"
    if not third_party_source.is_dir():
        raise FileNotFoundError(third_party_source)
    shutil.copytree(
        third_party_source,
        third_party_destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if args.source_copy is not None:
        shutil.copytree(
            package_root,
            args.source_copy,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.so"),
        )
    print(package_root)


if __name__ == "__main__":
    main()
