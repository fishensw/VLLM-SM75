# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for quantized DFlash context-KV and draft mappings.

Execute the shipped functions via AST so this suite also runs without CUDA or
an installed vLLM extension. NumPy supplies the tiny tensor operations; the
projection stand-in forbids raw packed-weight access and applies real scales.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "docker/speculative/vllm/model_executor/models"


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    @property
    def data(self):
        return self

    def __getitem__(self, index):
        return _Tensor(self.array[index])

    def __add__(self, other):
        return _Tensor(self.array + other.array)

    def __truediv__(self, scale):
        return _Tensor(self.array / scale)

    def view(self, *shape):
        return _Tensor(self.array.reshape(shape))

    def permute(self, *dims):
        return _Tensor(self.array.transpose(dims))

    def contiguous(self):
        return _Tensor(np.ascontiguousarray(self.array))


class _Unquantized:
    pass


def _linear(x, weight, bias=None):
    result = x.array @ weight.array.T
    if bias is not None:
        result = result + bias.array
    return _Tensor(result)


def _extract(filename, functions, methods=(), class_name="DFlashQwen3Model"):
    tree = ast.parse((MODELS / filename).read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in methods:
                    selected.append(method)
    assert {n.name for n in selected} == set(functions) | set(methods)
    # Delay annotations so these functions need no production imports.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    return compile(ast.fix_missing_locations(module), str(MODELS / filename), "exec")


@pytest.fixture
def runtime():
    torch = SimpleNamespace(
        cat=lambda tensors, dim=0: _Tensor(np.concatenate([t.array for t in tensors], axis=dim)),
        stack=lambda tensors, dim=0: _Tensor(np.stack([t.array for t in tensors], axis=dim)),
    )
    scope = {"torch": torch, "F": SimpleNamespace(linear=_linear), "UnquantizedLinearMethod": _Unquantized}
    methods = ("_build_context_kv_buffers", "_project_context_kv_per_layer", "_project_context_kv")
    exec(_extract("qwen3_dflash.py", ("_can_fuse_context_kv",), methods), scope)
    cls = type("ShippedDFlashMethods", (), {name: scope[name] for name in methods})
    return SimpleNamespace(scope=scope, model_cls=cls)


class _Projection:
    def __init__(self, weight, bias, *, quantized, return_bias=True, skip_bias_add=False):
        self.quant_method = object() if quantized else _Unquantized()
        # Exactly representable FP8-like values with a nontrivial tensor scale.
        self.checkpoint_weight = _Tensor(weight / 0.25)
        self.scale = 0.25
        self.dense_weight = _Tensor(weight)
        self.bias = None if bias is None else _Tensor(bias)
        self.return_bias = return_bias
        self.skip_bias_add = skip_bias_add
        self.calls = 0
        self.raw_reads = 0

    @property
    def weight(self):
        self.raw_reads += 1
        if not isinstance(self.quant_method, _Unquantized):
            raise AssertionError("packed FP8/Marlin weight must not be read directly")
        return self.dense_weight

    def __call__(self, x):
        self.calls += 1
        # Model the quantization method, including scale application.
        weight = _Tensor(self.checkpoint_weight.array * self.scale)
        output = _linear(x, weight, None if self.skip_bias_add else self.bias)
        if self.return_bias:
            return output, self.bias if self.skip_bias_add else None
        return output


def _build(runtime, modes, has_bias, *, return_bias=True, skip_bias_add=False):
    model = runtime.model_cls()
    model.hidden_norm = SimpleNamespace(weight=_Tensor(np.ones(4)))
    model.normalization_calls = 0

    def normalize(x):
        model.normalization_calls += 1
        # Ensure the shipped code keeps the DFlash2 normalization override.
        return _Tensor(x.array * 2)

    model._normalize_context_states = normalize
    projections, attns = [], []
    # Distinct rank-local Q widths catch hard-coded global slices under TP.
    for i, quantized in enumerate(modes):
        q_size = 2 + i
        weight = (np.arange((q_size + 4) * 4).reshape(q_size + 4, 4) % 7 - 3).astype(np.float32)
        bias = np.arange(q_size + 4).astype(np.float32) if has_bias else None
        projection = _Projection(weight, bias, quantized=quantized, return_bias=return_bias, skip_bias_add=skip_bias_add)
        projections.append(projection)
        attns.append(SimpleNamespace(qkv_proj=projection, q_size=q_size, k_norm=SimpleNamespace(weight=_Tensor(np.ones(2)))))
    model._build_context_kv_buffers(attns, has_bias)
    return model, projections, attns


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("modes", [(True, True), (False, True), (True, False)])
def test_quantized_kv_matches_dense_reference(runtime, has_bias, modes):
    fused, _, _ = _build(runtime, (False, False), has_bias)
    quantized, projections, _ = _build(runtime, modes, has_bias)
    x = _Tensor([[1, -2, 3, 1], [3, 1, -1, 2], [1, 2, 1, -3]])
    expected = fused._project_context_kv(x, 3, 2, 1, 2)
    actual = quantized._project_context_kv(x, 3, 2, 1, 2)
    assert quantized._fuse_context_kv is False
    assert not hasattr(quantized, "_fused_kv_weight")
    assert all(p.raw_reads == 0 and p.calls == 1 for p in projections)
    assert quantized.normalization_calls == 1
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result.array, reference.array)
        assert result.array.shape == (2, 3, 1, 2)
        assert result.array.flags.c_contiguous


@pytest.mark.parametrize("return_bias,skip_bias_add", [(True, True), (True, False), (False, False)])
def test_quantized_projection_preserves_bias_and_return_convention(runtime, return_bias, skip_bias_add):
    model, projections, attns = _build(runtime, (True, True), True, return_bias=return_bias, skip_bias_add=skip_bias_add)
    x = _Tensor([[1, 2, 3, 4]])
    result = model._project_context_kv_per_layer(x)
    expected = np.concatenate([_linear(x, p.dense_weight, p.bias).array[:, a.q_size:] for p, a in zip(projections, attns)], axis=-1)
    np.testing.assert_array_equal(result.array, expected)


def test_unquantized_projection_keeps_fused_path(runtime):
    model, projections, _ = _build(runtime, (False, False), True)
    assert model._fuse_context_kv is True
    assert all(p.raw_reads == 1 for p in projections)
    model._project_context_kv(_Tensor([[1, 2, 3, 4]]), 1, 2, 1, 2)
    assert all(p.calls == 0 for p in projections)


def test_rebuild_discards_stale_fused_weights(runtime):
    model, _, _ = _build(runtime, (False, False), False)
    _, _, quant_attns = _build(runtime, (True, True), False)
    model._build_context_kv_buffers(quant_attns, False)
    assert not hasattr(model, "_fused_kv_weight")
    assert not hasattr(model, "_fused_kv_bias")


@pytest.mark.parametrize("present,quantized", [(False, False), (True, False), (True, True)])
def test_draft_quant_config_configures_own_packed_mapping(monkeypatch, present, quantized):
    draft = object() if present else None
    load = object()
    config = SimpleNamespace() if quantized else None
    seen = []
    draft_cls = type("Draft", (), {"packed_modules_mapping": {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}})
    loader = ModuleType("vllm.model_executor.model_loader.utils")

    def get_architecture(model_config):
        assert model_config is draft
        seen.append("resolve")
        return draft_cls, "Draft"

    def configure(quant_config, model_class):
        assert quant_config is config and model_class is draft_cls
        quant_config.packed_modules_mapping = model_class.packed_modules_mapping
        seen.append("configure")

    loader.get_model_architecture = get_architecture
    loader.configure_quant_config = configure
    monkeypatch.setitem(sys.modules, loader.__name__, loader)

    def get_config(model_config, load_config):
        assert model_config is draft and load_config is load
        seen.append("get")
        return config

    scope = {"VllmConfig": SimpleNamespace(get_quantization_config=get_config)}
    exec(_extract("utils.py", ("get_draft_quant_config",)), scope)
    result = scope["get_draft_quant_config"](SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft), load_config=load))
    assert result is (config if present else None)
    assert seen == (["get", "resolve", "configure"] if present and quantized else ["get"] if present else [])
    if present and quantized:
        assert config.packed_modules_mapping == draft_cls.packed_modules_mapping



def test_sm75_bf16_mlp_preserves_quantized_module_calls():
    """The real issue-13 FP8 draft quantizes MLP, while QKV is ignored."""
    scope = {"DFLASH_SM70_GATE_UP_INPUT_SCALE": 32.0}
    intermediates = {}

    def silu_and_mul(gate_up):
        intermediates["gate_up"] = gate_up.array.copy()
        gate, up = np.split(gate_up.array * 32.0, 2, axis=-1)
        activated = gate / (1.0 + np.exp(-gate)) * up
        return _Tensor(activated / 4.0), _Tensor(np.array([[4.0]]))

    def scale_output(value, scales):
        return _Tensor(value.array * scales.array / 256.0)

    scope["dflash_silu_and_mul_sm70"] = silu_and_mul
    scope["dflash_scale_output_sm70"] = scale_output
    exec(_extract(
        "dflash_sm70.py", (), ("forward",), class_name="DFlashSM70MLP"
    ), scope)
    gate_weight = np.arange(24, dtype=np.float32).reshape(6, 4) / 64
    down_weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 64
    gate = _Projection(gate_weight, None, quantized=True)
    down = _Projection(down_weight, None, quantized=True)
    model = SimpleNamespace(gate_up_proj=gate, down_proj=down)
    x = _Tensor([[1.0, 2.0, -1.0, 0.5]])
    result = scope["forward"](model, x)
    projected = x.array @ gate_weight.T
    gate_ref, up_ref = np.split(projected, 2, axis=-1)
    expected = (gate_ref / (1 + np.exp(-gate_ref)) * up_ref) @ down_weight.T
    np.testing.assert_allclose(result.array * 256.0, expected, rtol=1e-6)
    np.testing.assert_allclose(intermediates["gate_up"] * 32, projected)
    assert gate.calls == down.calls == 1
    assert gate.raw_reads == down.raw_reads == 0



@pytest.mark.parametrize("quantized_fc", [False, True])
@pytest.mark.parametrize("tp_size", [1, 4])
def test_partial_quantization_keeps_unquantized_context_fc_sharding(
    quantized_fc, tp_size
):
    scope = {
        "_use_sm75_bf16_emulation": lambda: True,
        "get_tensor_model_parallel_world_size": lambda: tp_size,
        "UnquantizedLinearMethod": _Unquantized,
    }
    exec(_extract("qwen3_dflash2.py", ("_use_sm75_tp4_context_fc",)), scope)
    fc = SimpleNamespace(
        input_size=25600,
        quant_method=object() if quantized_fc else _Unquantized(),
    )
    use_sharding = scope["_use_sm75_tp4_context_fc"]
    assert use_sharding(fc, SimpleNamespace(hidden_size=5120), True) is (
        not quantized_fc and tp_size == 4
    )
    assert not use_sharding(fc, SimpleNamespace(hidden_size=5120), False)
    assert not use_sharding(fc, SimpleNamespace(hidden_size=4096), True)
    fc.input_size = 20480
    assert not use_sharding(fc, SimpleNamespace(hidden_size=5120), True)
