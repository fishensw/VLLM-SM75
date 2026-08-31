# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.third_party.flash_qla_sm75 import fused_fwd


class _FakeCudaPlatform:
    def __init__(self, capability: tuple[int, int] = (7, 5)) -> None:
        self.capability = capability

    def is_cuda(self) -> bool:
        return True

    def is_device_capability(self, capability: int) -> bool:
        return self.capability == divmod(capability, 10)

    def is_device_capability_family(self, capability: int) -> bool:
        return self.capability[0] == capability // 10

    def get_cuda_runtime_major(self) -> int:
        return 12


@pytest.fixture(autouse=True)
def reset_extension_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fused_fwd, "_EXT", None)
    monkeypatch.setattr(fused_fwd, "_EXT_LOAD_ERROR", None)


def _config(
    *,
    backend: str = "flashqla_sm75",
    dtype: torch.dtype = torch.float16,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
) -> SimpleNamespace:
    text_config = SimpleNamespace(
        linear_key_head_dim=head_k_dim,
        linear_value_head_dim=head_v_dim,
    )
    model_config = SimpleNamespace(hf_text_config=text_config, dtype=dtype)
    return SimpleNamespace(
        additional_config={"gdn_prefill_backend": backend},
        model_config=model_config,
        speculative_config=None,
    )


@pytest.mark.parametrize(
    "config,capability",
    [
        (_config(dtype=torch.bfloat16), (7, 5)),
        (_config(head_k_dim=64), (7, 5)),
        (_config(head_v_dim=64), (7, 5)),
        (_config(), (8, 0)),
    ],
)
def test_flashqla_sm75_prefill_falls_back_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    config: SimpleNamespace,
    capability: tuple[int, int],
) -> None:
    monkeypatch.setattr(gdn, "current_platform", _FakeCudaPlatform(capability))

    requested, active = gdn._resolve_gdn_prefill_backend(config)

    assert requested == "flashqla_sm75"
    assert active == "triton"


def test_flashqla_sm75_prefill_is_explicit_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdn, "current_platform", _FakeCudaPlatform())

    assert gdn._resolve_gdn_prefill_backend(_config()) == (
        "flashqla_sm75",
        "flashqla_sm75",
    )
    assert gdn._resolve_gdn_prefill_backend(_config(backend="auto")) == (
        "auto",
        "triton",
    )


def test_sm75_loader_disables_runtime_jit_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_prebuilt(name: str):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(fused_fwd.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fused_fwd.importlib, "import_module", missing_prebuilt)
    monkeypatch.delenv(fused_fwd._PREBUILT_EXTENSION_PATH_ENV, raising=False)
    monkeypatch.delenv(fused_fwd._ALLOW_JIT_ENV, raising=False)
    monkeypatch.setattr(
        fused_fwd,
        "load",
        lambda **kwargs: pytest.fail("runtime JIT must remain disabled"),
    )

    with pytest.raises(RuntimeError, match="runtime JIT is disabled"):
        fused_fwd._load_ext()


def test_sm75_loader_fails_closed_for_missing_image_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    missing = tmp_path / "flash_qla_sm75_gdn.so"
    monkeypatch.setattr(fused_fwd.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv(fused_fwd._PREBUILT_EXTENSION_PATH_ENV, str(missing))
    monkeypatch.delenv(fused_fwd._ALLOW_JIT_ENV, raising=False)
    monkeypatch.setattr(
        fused_fwd,
        "load",
        lambda **kwargs: pytest.fail("runtime JIT must remain disabled"),
    )

    with pytest.raises(RuntimeError, match="Failed to load"):
        fused_fwd._load_ext()


def test_sm75_development_jit_emits_only_sm75(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = ModuleType("flash_qla_sm75_gdn")
    load_kwargs = {}

    def missing_prebuilt(name: str):
        raise ModuleNotFoundError(name=name)

    def fake_load(**kwargs):
        load_kwargs.update(kwargs)
        return loaded

    monkeypatch.setattr(fused_fwd.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fused_fwd.importlib, "import_module", missing_prebuilt)
    monkeypatch.delenv(fused_fwd._PREBUILT_EXTENSION_PATH_ENV, raising=False)
    monkeypatch.setenv(fused_fwd._ALLOW_JIT_ENV, "1")
    monkeypatch.setattr(fused_fwd, "load", fake_load)

    assert fused_fwd._load_ext() is loaded
    assert load_kwargs["extra_cuda_cflags"] == [
        "-O3",
        "-gencode=arch=compute_75,code=sm_75",
    ]
