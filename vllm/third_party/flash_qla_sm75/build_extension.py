# SPDX-License-Identifier: Apache-2.0
"""Build the vendored FlashQLA extension for SM75 only."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    source = Path(__file__).with_name("csrc") / "gdn_forward.cu"
    args.build_directory.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extension = load(
        name="flash_qla_sm75_gdn",
        sources=[str(source)],
        build_directory=str(args.build_directory),
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_75,code=sm_75",
        ],
        extra_cflags=["-O3"],
        verbose=args.verbose,
    )
    extension_path = Path(extension.__file__).resolve()
    shutil.copy2(extension_path, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
