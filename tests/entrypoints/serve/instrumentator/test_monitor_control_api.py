"""The Ultra dashboard can hide the standard page and retain its control API."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def monitor():
    source = Path(__file__).resolve().parents[4] / "vllm/entrypoints/serve/instrumentator/monitor.py"
    envs = ModuleType("vllm.envs")
    envs.VLLM_MONITOR = False
    envs.VLLM_API_KEY = ""
    package = ModuleType("vllm")
    package.envs = envs
    spec = importlib.util.spec_from_file_location("monitor_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"vllm": package, "vllm.envs": envs}):
        spec.loader.exec_module(module)
    return module


def make_client(monitor, *, api_key=None, configured=True):
    app = FastAPI()
    app.state.args = SimpleNamespace(api_key=[api_key] if api_key else [])
    engine = SimpleNamespace(
        is_speculative_decoding_configured=AsyncMock(return_value=configured),
        is_speculative_decoding_enabled=AsyncMock(return_value=True),
        set_speculative_decoding=AsyncMock(),
    )
    app.state.engine_client = engine
    monitor.attach_router(app)
    return TestClient(app), engine


def test_hidden_page_preserves_control_routes(monitor):
    client, engine = make_client(monitor)
    assert client.get("/monitor").status_code == 404
    assert client.get("/monitor/spec_decode").json() == {
        "spec_configured": True, "enabled": True, "auth_required": False,
    }
    response = client.post("/monitor/spec_decode", json={"enabled": False})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "enabled": False}
    engine.set_speculative_decoding.assert_awaited_once_with(False)


def test_hidden_page_control_api_still_requires_configured_key(monitor):
    client, engine = make_client(monitor, api_key="test-key")
    response = client.post("/monitor/spec_decode", json={"enabled": False})
    assert response.status_code == 401
    engine.set_speculative_decoding.assert_not_awaited()
    response = client.post("/monitor/spec_decode", json={"enabled": False}, headers={
        "x-api-key-hash": hashlib.sha256(b"test-key").hexdigest(),
    })
    assert response.status_code == 200
    engine.set_speculative_decoding.assert_awaited_once_with(False)


def test_unconfigured_speculation_cannot_be_enabled(monitor):
    client, engine = make_client(monitor, configured=False)
    assert client.get("/monitor/spec_decode").json()["enabled"] is False
    assert client.post("/monitor/spec_decode", json={"enabled": True}).status_code == 409
    engine.set_speculative_decoding.assert_not_awaited()


def test_enabled_page_remains_available(monitor):
    monitor.envs.VLLM_MONITOR = True
    client, _ = make_client(monitor)
    response = client.get("/monitor")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
