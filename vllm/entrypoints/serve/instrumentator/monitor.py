# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""自包含监控看板: serve 在 /monitor 返回本目录 dashboard.html 单文件页。

页面纯前端(无 CDN/无外部依赖, 图表用手写 canvas 实现), 轮询同源 /metrics
(Prometheus 文本)渲染并发/KV 趋势、prefix 缓存命中、token 统计、延迟分位
(P50/P90/P99)、prompt/generation token 分布、preemption 与 sleep 状态。
HTML 独立成 dashboard.html, 可直接用浏览器打开看样式; 每次请求读盘, 改样式
无需重启。VLLM_MONITOR 默认开, '0'/'off'/'false'/'no' 关(不挂路由)。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from vllm import envs

# 看板 HTML(独立文件, 便于直接打开/编辑), 与 monitor.py 同目录。
_DASHBOARD_HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"


def attach_router(app: FastAPI) -> None:
    """按 VLLM_MONITOR 开关把 /monitor 挂到 app。

    关(0/off/false/no)时不挂任何路由。/monitor 不在 GUARDED_PREFIX 内,
    浏览器无需 API key 即可访问; 页面内 fetch /metrics 亦同源无鉴权。
    每次请求读盘 dashboard.html, 改样式直接刷新即可(无需重启)。
    """
    if not envs.VLLM_MONITOR:
        return

    @app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
    def monitor() -> HTMLResponse:  # noqa: N802
        return HTMLResponse(_DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))
