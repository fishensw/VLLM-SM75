"""Reject incompatible fast runtimes before modifying the installed package."""
import importlib.metadata
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("version", ["0.29.0+cu129", "0.31.0", "0.30.1"])
def test_wrong_version_leaves_package_untouched(tmp_path, monkeypatch, version):
    package = tmp_path / "vllm"
    package.mkdir()
    marker = package / "__init__.py"
    marker.write_text("original package\n")
    module = ModuleType("vllm")
    module.__file__ = str(marker)
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(sys, "argv", ["install_sm75_overlay.py", str(ROOT / "vllm")])
    main = runpy.run_path(str(ROOT / "docker/helpers/install_sm75_overlay.py"))["main"]
    with pytest.raises(RuntimeError, match="require vLLM 0.30.0"):
        main()
    assert list(package.iterdir()) == [marker]
    assert marker.read_text() == "original package\n"
