# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""构建期编译 firefly fused fp8 GEMM 扩展(仅 sm_75), 免运行时 JIT。

镜像 build 时由 firefly-fused-builder stage 调用, 产物 firefly_fused.so 供
runtime 直接 importlib 加载(见 firefly.py _load_fused_mod 的 prebuilt 分支)。
include = 已装 flashinfer 的 cutlass(2.x + cute + fp8) + 本目录 _cutlass_ext
(打包的 vllm cutlass_extensions 头)。仿 third_party/flash_qla_sm75/build_extension.py。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import flashinfer
from torch.utils.cpp_extension import load


def _find_cutlass_include() -> Path:
    fi = (
        Path(flashinfer.__file__).resolve().parent / "data" / "cutlass" / "include"
    )
    assert fi.is_dir(), f"flashinfer cutlass include not found at {fi}"
    return fi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    source = here / "firefly_fused.cu"
    cutlass_inc = _find_cutlass_include()
    cutlass_ext_inc = here / "_cutlass_ext"
    assert source.is_file(), f"firefly_fused.cu not found at {source}"
    assert cutlass_ext_inc.is_dir(), f"_cutlass_ext not found at {cutlass_ext_inc}"

    args.build_directory.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extension = load(
        name="firefly_fused",
        sources=[str(source)],
        build_directory=str(args.build_directory),
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_75,code=sm_75"],
        extra_include_paths=[str(cutlass_inc), str(cutlass_ext_inc)],
        verbose=args.verbose,
    )
    extension_path = Path(extension.__file__).resolve()
    shutil.copy2(extension_path, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
