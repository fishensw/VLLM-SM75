"""Exercise the SM75 Firefly adapter against the v0.30 Marlin API without CUDA.

The overlay is not a complete importable vLLM checkout. Extract the production
functions and execute them with strict boundary doubles, so a removed upstream
argument or attribute fails on machines without PyTorch as well.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "vllm/model_executor/kernels/linear/mixed_precision/marlin.py"


class Tensor:
    dtype = "float16"
    device = "cuda"

    def __init__(self, shape=(1, 128)):
        self.shape = shape
        self.data = self

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total


def load_functions(namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_ff_hybrid_linear":
            node.decorator_list = []
            body.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "MarlinLinearKernel":
            body.extend(method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in {"_firefly_enabled", "apply_weights"})
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


@pytest.fixture
def api():
    result = Tensor((1, 64))

    # This is the strict v0.30 signature: no g_idx, sorting permutation or
    # is_k_full parameters. Unknown legacy keywords raise TypeError.
    def marlin(*, input, weight, weight_scale, weight_zp, workspace, wtype,
               input_size_per_partition, output_size_per_partition,
               input_global_scale=None, bias=None, input_dtype=None):
        assert input_size_per_partition == 128
        assert output_size_per_partition == 64
        return result

    namespace = {
        "torch": SimpleNamespace(float16="float16", bfloat16="bfloat16", empty=lambda *a, **k: Tensor((0,))),
        "scalar_types": SimpleNamespace(uint4="uint4", uint4b8="uint4b8"),
        "firefly_active_int4": lambda: True,
        "envs": SimpleNamespace(VLLM_FIREFLY_MIN_M=16, VLLM_FIREFLY_DEQUANT_MODEL="fast"),
        "apply_gptq_marlin_linear": Mock(side_effect=marlin),
        "dequant_marlin_to_int8_cached": Mock(return_value="int8-weights"),
        "int8_prefill_linear": Mock(return_value=result),
        "result": result,
    }
    return load_functions(namespace)


def config(**changes):
    values = dict(weight_type="uint4b8", act_type="float16", zero_points=False,
                  partition_weight_shape=(128, 64))
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("legacy,enabled", [(None, True), (False, True), (True, False)])
def test_gate_accepts_v030_config_without_has_g_idx(api, legacy, enabled):
    cfg = config()
    if legacy is not None:
        cfg.has_g_idx = legacy
    assert api["_firefly_enabled"](SimpleNamespace(config=cfg)) is enabled


def invoke_hybrid(api, x):
    return api["_ff_hybrid_linear"](
        x=x, w_q=Tensor(), w_s_marlin=Tensor(), w_zp_marlin=Tensor((0,)),
        workspace=Tensor(), input_global_scale=Tensor((0,)), bias=Tensor((0,)),
        w_s_clean=Tensor(), w_zp_clean=Tensor((0,)), c_n=Tensor(),
        size_k=128, size_n=64, padded_k=128, padded_n=64, gs=128, min_m=16,
        has_zp=False, use_recip=True,
    )


def test_hybrid_decode_calls_v030_marlin_signature(api):
    assert invoke_hybrid(api, Tensor((1, 128))) is api["result"]
    api["apply_gptq_marlin_linear"].assert_called_once()
    api["dequant_marlin_to_int8_cached"].assert_not_called()


def test_hybrid_prefill_keeps_int8_dispatch(api):
    x = Tensor((32, 128))
    assert invoke_hybrid(api, x) is api["result"]
    api["apply_gptq_marlin_linear"].assert_not_called()
    api["int8_prefill_linear"].assert_called_once()
    assert api["dequant_marlin_to_int8_cached"].call_args.kwargs == {"w_zp": None, "use_recip": True}


@pytest.mark.parametrize("enabled", [False, True])
def test_apply_weights_uses_three_weight_tuple_without_legacy_fields(api, enabled):
    layer = SimpleNamespace(_firefly_ws=Tensor(), _firefly_c_n=Tensor(),
                            _firefly_padded_k=128, _firefly_padded_n=64,
                            _firefly_gs=128)
    kernel = SimpleNamespace(config=config(), workspace=Tensor(),
                             _firefly_enabled=lambda: enabled,
                             _get_weight_params=lambda layer: (Tensor(), Tensor(), Tensor((0,))))
    assert api["apply_weights"](kernel, layer, Tensor()) is api["result"]
    api["apply_gptq_marlin_linear"].assert_called_once()
