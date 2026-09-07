# SPDX-License-Identifier: Apache-2.0
"""Verify and record the fixed runtime contract for the SM75 image."""

from __future__ import annotations

import ast
import importlib.metadata
import json
import tempfile
from pathlib import Path

import torch

EXPECTED_PACKAGES = {
    "vllm": "0.28.0",
    "flashinfer-python": "0.6.18",
    "flashinfer-cubin": "0.6.18",
    "transformers": "5.15.1",
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

    from vllm.config.model import _normalize_config_dtype
    from vllm.entrypoints.serve.utils.api_utils import _redact_sensitive_args

    redacted = _redact_sensitive_args(
        {"api_key": ["build-secret"], "hf_token": "build-token", "model": "ok"}
    )
    if redacted != {
        "api_key": "<REDACTED>",
        "hf_token": "<REDACTED>",
        "model": "ok",
    }:
        raise RuntimeError(f"Sensitive argument redaction failed: {redacted}")
    if _normalize_config_dtype("float16") is not torch.float16:
        raise RuntimeError("String HF dtype override was not normalized")

    from vllm.v1.engine.auto_sleep import AutoSleepConfig, AutoSleepController

    if AutoSleepConfig(timeout_seconds=60.0).sleep_level != 1:
        raise RuntimeError("auto-sleep 'cpu' target must map to sleep level 1")
    if (
        AutoSleepConfig(
            timeout_seconds=60.0, offload_target="reload", reload_path="/ckpt"
        ).sleep_level
        != 2
    ):
        raise RuntimeError("auto-sleep 'reload' target must map to sleep level 2")
    if AutoSleepController(object(), object()).enabled:
        raise RuntimeError(
            "auto-sleep must be disabled without VLLM_AUTO_SLEEP_* envs"
        )
    from vllm.v1.engine.auto_sleep import warm_safetensors_page_cache

    if (
        AutoSleepConfig(timeout_seconds=60.0).page_cache_keep_interval_seconds
        != 600.0
    ):
        raise RuntimeError("auto-sleep page-cache keeper default must be 600s")
    if not AutoSleepConfig(timeout_seconds=60.0, offload_target="exit").is_exit:
        raise RuntimeError("auto-sleep must recognize the 'exit' offload target")
    with tempfile.TemporaryDirectory() as tmp:
        if warm_safetensors_page_cache(tmp) != 0:
            raise RuntimeError(
                "page-cache warm on an empty dir must advise 0 files"
            )

    # Deep-sleep exit/respawn plumbing (engine-core proc + client side).
    # Checked statically with ast against the installed source copy instead
    # of importing vllm.v1.engine.{core,core_client,async_llm}: those
    # imports pull in the whole engine stack (executor, model executor,
    # attention backends) and add several GiB of peak memory to the build,
    # on top of the parallel nvcc stage.
    engine_source = Path("/opt/vllm-sm75/source/v0.1.0/vllm/v1/engine")

    def _class_members(path: Path) -> dict[str, set[str]]:
        tree = ast.parse(path.read_text(), filename=str(path))
        classes: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            members: set[str] = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.Assign):
                    members.update(
                        target.id
                        for target in item.targets
                        if isinstance(target, ast.Name)
                    )
            classes[node.name] = members
        return classes

    core_classes = _class_members(engine_source / "core.py")
    core_members = core_classes.get("EngineCoreProc", set())
    if "DEEP_SLEEP_EXITING" not in core_members:
        raise RuntimeError("EngineCoreProc must define DEEP_SLEEP_EXITING")
    if "request_deep_sleep_exit" not in core_members:
        raise RuntimeError("EngineCoreProc must define request_deep_sleep_exit")

    client_classes = _class_members(engine_source / "core_client.py")
    if "_respawn_launch" not in client_classes.get("MPClient", set()):
        raise RuntimeError("MPClient must define _respawn_launch")
    if "respawn_engine" not in client_classes.get("AsyncMPClient", set()):
        raise RuntimeError("AsyncMPClient must define respawn_engine")

    async_llm_classes = _class_members(engine_source / "async_llm.py")
    if "_await_deep_sleep_respawn" not in async_llm_classes.get("AsyncLLM", set()):
        raise RuntimeError("AsyncLLM must define _await_deep_sleep_respawn")

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
