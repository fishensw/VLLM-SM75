# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""自包含监控看板: serve 在 /monitor 返回本目录 dashboard.html 单文件页。

页面纯前端(无 CDN/无外部依赖, 图表用手写 canvas 实现), 轮询同源 /metrics
(Prometheus 文本)渲染并发/KV 趋势、prefix 缓存命中、token 统计、延迟分位
(P50/P90/P99)、prompt/generation token 分布、preemption 与 sleep 状态。
HTML 独立成 dashboard.html, 可直接用浏览器打开看样式; 每次请求读盘, 改样式
无需重启。VLLM_MONITOR 默认开, '0'/'off'/'false'/'no' 关(不挂路由)。
"""

import hashlib
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from vllm import envs

# 看板 HTML(独立文件, 便于直接打开/编辑), 与 monitor.py 同目录。
_DASHBOARD_HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"

# 关值集合(移植到其他 vllm 时, 若其 envs 无 VLLM_MONITOR 字段, 回退直读该 env)。
_OFF_VALUES = ("0", "off", "false", "no")


def _monitor_enabled() -> bool:
    """VLLM_MONITOR 开关: 走 envs 归一化; 移植到无此字段的 vllm 时
    (envs.VLLM_MONITOR 抛 AttributeError), 回退直读环境变量, 未设默认开。"""
    try:
        return bool(envs.VLLM_MONITOR)
    except AttributeError:
        return os.environ.get("VLLM_MONITOR", "1").strip().lower() not in _OFF_VALUES


def _configured_api_keys(request: Request) -> list[str]:
    """取配置的 API key 列表(CLI --api-key 优先, VLLM_API_KEY 兜底)。

    与 api_server.py 取 token 的逻辑一致: args.api_key 为空时回退 env。
    """
    args = getattr(request.app.state, "args", None)
    cli_keys = getattr(args, "api_key", None) if args is not None else None
    if cli_keys:
        return [k for k in cli_keys if k]
    try:
        env_key = envs.VLLM_API_KEY
    except AttributeError:
        env_key = os.environ.get("VLLM_API_KEY", "")
    return [env_key] if env_key else []


def _verify_api_key(request: Request) -> bool:
    """校验请求头 x-api-key-hash 是否匹配任一配置 key 的 SHA-256。

    未配置任何 --api-key 时直接放行(与 vLLM 无 key 时全开放一致)。
    前端把 API key 哈希成 hex 再传(原 key 不裸传), 这里对每个配置 key 算
    SHA-256 后 compare_digest 比对, 与 authenticate.py 同一套手法。
    """
    keys = _configured_api_keys(request)
    if not keys:
        return True
    digest = request.headers.get("x-api-key-hash", "").strip().lower()
    if not digest:
        return False
    for key in keys:
        if secrets.compare_digest(digest, hashlib.sha256(key.encode("utf-8")).hexdigest()):
            return True
    return False


def attach_router(app: FastAPI) -> None:
    """按 VLLM_MONITOR 开关把 /monitor 挂到 app。

    关(0/off/false/no)时不挂任何路由。/monitor 不在 GUARDED_PREFIX 内,
    浏览器无需 API key 即可访问; 页面内 fetch /metrics 亦同源无鉴权。
    每次请求读盘 dashboard.html, 改样式直接刷新即可(无需重启)。

    额外挂投机解码开关:
    - GET  /monitor/spec_decode  只读, 无鉴权, 返回 {spec_configured, enabled}
    - POST /monitor/spec_decode  写, 需 x-api-key-hash(SHA-256), 切换开关
    开关只跳草稿计算, 不卸显存; 未配 --speculative-config 时 POST 返回 409。
    """
    if not _monitor_enabled():
        return

    @app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
    def monitor() -> HTMLResponse:  # noqa: N802
        return HTMLResponse(_DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))

    @app.get("/monitor/spec_decode", include_in_schema=False)
    async def get_spec_decode(request: Request) -> JSONResponse:  # noqa: N802
        engine = request.app.state.engine_client
        configured = await engine.is_speculative_decoding_configured()
        enabled = (
            await engine.is_speculative_decoding_enabled() if configured else False
        )
        return JSONResponse(
            {
                "spec_configured": configured,
                "enabled": enabled,
                # 无 --api-key 时为 False, 前端据此免弹 key 输入框。
                "auth_required": bool(_configured_api_keys(request)),
            }
        )

    @app.post("/monitor/spec_decode", include_in_schema=False)
    async def set_spec_decode(request: Request) -> JSONResponse:  # noqa: N802
        if not _verify_api_key(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 - 空/坏 body 一律当 {enabled:true}
            payload = {}
        enabled = bool(payload.get("enabled", True))
        engine = request.app.state.engine_client
        if not await engine.is_speculative_decoding_configured():
            return JSONResponse(
                {"error": "speculative decoding not configured"}, status_code=409
            )
        await engine.set_speculative_decoding(enabled)
        return JSONResponse({"ok": True, "enabled": enabled})
