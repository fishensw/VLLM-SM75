# SPDX-License-Identifier: Apache-2.0
"""Apply fail-closed fixes for Transformers 5.15.1 auto-docstring noise."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

PACKAGE_ROOT = Path("/usr/local/lib/python3.12/dist-packages/transformers")


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one match in {path}: {old!r}")
    path.write_text(source.replace(old, new))


def main() -> None:
    version = importlib.metadata.version("transformers")
    if version != "5.15.1":
        raise RuntimeError(f"Expected transformers 5.15.1, found {version}")

    for relative in (
        "models/deepseek_vl_hybrid/image_processing_deepseek_vl_hybrid.py",
        "models/deepseek_vl_hybrid/image_processing_pil_deepseek_vl_hybrid.py",
    ):
        replace_once(
            PACKAGE_ROOT / relative,
            "     high_res_size (`dict`, *optional*",
            "    high_res_size (`dict`, *optional*",
        )

    replace_once(
        PACKAGE_ROOT / "models/qwen3_vl/video_processing_qwen3_vl.py",
        """    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    \"\"\"""",
        """    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    min_frames (`int`, *optional*, defaults to 4):
        The minimum number of frames sampled from a video.
    max_frames (`int`, *optional*, defaults to 768):
        The maximum number of frames sampled from a video.
    \"\"\"""",
    )


if __name__ == "__main__":
    main()
