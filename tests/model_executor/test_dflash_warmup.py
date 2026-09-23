"""Bound DFlash warmup for small token budgets and memory-only profiling."""
import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "docker/speculative/vllm/v1/worker/gpu/model_runner.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_dflash_warmup_batch_size")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
batch_size = namespace[function.name]


@pytest.mark.parametrize("requests,tokens,width,profile,expected", [
    (256, 512, 8, False, 64),
    (4, 8192, 8, False, 4),
    (128, 1025, 8, False, 128),
    (4, 7, 8, False, 0),
    (4, 8192, 8, True, 0),
    (4, 8192, 0, False, 0),
])
def test_warmup_respects_capacity_and_profile(requests, tokens, width, profile, expected):
    result = batch_size(requests, tokens, width, profile_only=profile)
    assert result == expected
    assert result * width <= tokens
