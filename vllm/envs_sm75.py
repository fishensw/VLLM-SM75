# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM75 扩展 env —— 对上游 vllm/envs.py 的纯增量, 不再整文件覆盖。

底座保留上游原样 vllm/envs.py; 本模块只装 SM75 自定义的 env getter,
构建时由 install_sm75_overlay.py 把一行 `import vllm.envs_sm75; apply()`
append 到上游 envs.py 尾部触发注入。注入做两件事:

1) 把 EXTENSIONS getter 灌进 vllm.envs.environment_variables —— 上游
   __getattr__ / __dir__ / is_set / validate_environ / enable_envs_cache
   全围绕该 dict, 灌入后 envs.VLLM_FIREFLY 等属性访问自动生效(无需改调用方)。
2) 包一层 vllm.envs.compile_factors, 把 INSTALL_IGNORED(idle auto-sleep 计时器)
   从 hash factors pop 掉 —— 它们只影响调度/checkpoint, 不改变编译图。

好处: 上游 envs.py 后续升级随便改(内容级), 本文件只跟「dict + __getattr__
+ compile_factors 返回 dict」这个稳定机制耦合, 冲突面从整文件缩到一处注入。
"""

import os

# 需从 compile_factors hash 中排除的 key: idle auto-sleep 计时器/路径只影响
# 调度与 checkpoint bookkeeping, 不改变编译图。切 test/prod sleep 计时不应
# 使 torch.compile 缓存失配。
INSTALL_IGNORED = {
    "VLLM_AUTO_SLEEP_IDLE_TIMEOUT",
    "VLLM_AUTO_SLEEP_OFFLOAD_TARGET",
    "VLLM_AUTO_SLEEP_RELOAD_PATH",
    "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL",
}


def _firefly_mode() -> str:
    """VLLM_FIREFLY 归一化: '1'=开(=auto) / '0'=关。

    未设默认 '0'(不激活); '1' 与 'auto' 等价(都算开); 其余值一律 '0'。
    开 = int4(AWQ/GPTQ) 走 int8 加速, fp8 走上游 marlin(fp8 加速走
    VLLM_FIREFLY_AR fp8 allreduce, 另见 PLAN-fp8-allreduce)。
    """
    v = os.getenv("VLLM_FIREFLY", "").strip().lower()
    return "1" if v in ("1", "auto", "on", "true", "yes") else "0"


def _firefly_ar_mode() -> str:
    """VLLM_FIREFLY_AR 归一化: 'auto'(默认, 跟随 VLLM_FIREFLY) / '0'=强制关 /
    'fp8'=强制开。

    auto = firefly 模式开 (VLLM_FIREFLY=1) 时 fp8 allreduce 自动启用 (fp8 近乎
    无损, PLAN-fp8-allreduce §3.5); firefly 关 → AR 关。'fp8' 单独开 (firefly
    不用也开 AR); '0' 单独关 (firefly 开但 AR 不用)。FireflyAllReduce 只做 fp8,
    双 backend (P2P / SHM, 见 VLLM_FIREFLY_AR_BACKEND)。
    """
    v = os.getenv("VLLM_FIREFLY_AR", "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "0"
    if v in ("fp8", "1", "on", "true", "yes"):
        return "fp8"
    return "auto"


def _firefly_ar_backend() -> str:
    """VLLM_FIREFLY_AR_BACKEND 归一化: 'auto'(默认, 运行时按 _can_p2p 选 P2P
    优先) / 'p2p'(强制 P2P) / 'shm'(强制 SHM)。

    auto = 有 P2P (NVLink/PCIe 直连) 选 P2P (data/flag 全 device 显存, 无 host
    bounce); 无 P2P (PHB, 如 T10) 回 SHM (/dev/shm + cudaHostRegister)。
    'p2p'/'shm' 强制指定 (P2P 初始化失败仍自动回 SHM 兜底)。
    """
    v = os.getenv("VLLM_FIREFLY_AR_BACKEND", "").strip().lower()
    if v in ("p2p",):
        return "p2p"
    if v in ("shm", "host", "shared"):
        return "shm"
    return "auto"


def _monitor() -> bool:
    """VLLM_MONITOR 归一化: 默认开; '0'/'off'/'false'/'no' 关。

    开 = serve 在 /monitor 挂单文件 HTML 监控页(自拉同源 /metrics 渲染, 无 CDN,
    见 entrypoints/serve/instrumentator/monitor.py); 关 = 不挂该路由。
    """
    return os.getenv("VLLM_MONITOR", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


# SM75 自定义 env → getter。install 时灌进 vllm.envs.environment_variables,
# 之后 envs.<NAME> 属性访问 / is_set / validate_environ / __dir__ 自动生效。
EXTENSIONS: dict[str, object] = {
    # firefly(SM75) prefill 加速总开关, 默认关(_firefly_mode 归一化):
    #   未设 / 0 = 关(全走上游 marlin, 默认不激活)。
    #   1 / auto = 开(两者等价): int4(AWQ/GPTQ, W4A16) 走 int8 加速; fp8 走上游
    #     marlin(sm75 实测 firefly-fp8 不比 marlin 快, 已移除; fp8 加速改走
    #     VLLM_FIREFLY_AR fp8 allreduce, 见 PLAN-fp8-allreduce)。
    # 大 M 现反量化成 int8 走 CUTLASS(IMMA), 小 M/decode 保持 marlin。见
    # model_executor/layers/quantization/utils/firefly.py。
    "VLLM_FIREFLY": _firefly_mode,
    # 默认 1024: T10 上 p3_perf_sweep(6144x5120 层)测得 int8 反量化 ~1ms/层
    # (M 无关地板), crossover M≈854; M>1024 int8 才稳定快于 marlin(1.13-1.33x)。
    "VLLM_FIREFLY_MIN_M": lambda: int(
        os.environ.get("VLLM_FIREFLY_MIN_M", "1024")
    ),
    # int8 prefill 反量化量化步: "def"(默认) 除法(与两遍 pass2 逐 bit 一致);
    # "fast" 乘倒数(per-row r=1/c_n, 再省 ~30% 反量化, off-by-one ≤0.06%)。
    "VLLM_FIREFLY_DEQUANT_MODEL": lambda: (
        os.environ.get("VLLM_FIREFLY_DEQUANT_MODEL", "def")
    ),
    # fp8 allreduce (FireflyAllReduce, SHM backend, 无 P2P 如 T10)。auto(默认)
    # 跟随 VLLM_FIREFLY (firefly 开→AR 自动开); fp8 单独开; 0 单独关。TP2 每层 2 次
    # AllReduce 量减半 (fp16->fp8), 省 ~480-530ms/27B prefill。见
    # distributed/device_communicators/firefly_allreduce.py / PLAN-fp8-allreduce。
    "VLLM_FIREFLY_AR": _firefly_ar_mode,
    # 只对大消息走 FireflyAllReduce, 小消息回退 NCCL。decode 小消息 (几百 KB)
    # 时 firefly 的 amax 扫描 + 多轮 flag spin 固定开销可能超过砍半省下的传输
    # 时间 → 反而更慢。默认 1MB (fp16 字节): 27B decode M=1 单 AR 8KB << 1MB,
    # 走 NCCL; prefill M>=256 (2MB) 起走 firefly。见 firefly_allreduce.py。
    "VLLM_FIREFLY_AR_MIN_SIZE": lambda: int(
        os.environ.get("VLLM_FIREFLY_AR_MIN_SIZE", "1048576")
    ),
    # FireflyAllReduce 传输 backend: auto(默认, 运行时 _can_p2p 选 P2P 优先) /
    # p2p(强制) / shm(强制)。P2P 有 NVLink/PCIe 直连时 data/flag 全 device 显存
    # 无 host bounce; 无 P2P (PHB 如 T10) 回 SHM。见 firefly_allreduce.py。
    "VLLM_FIREFLY_AR_BACKEND": _firefly_ar_backend,
    # 单文件 HTML 监控页开关, 默认开(_monitor 归一化)。
    # 开 = serve 在 /monitor 挂自包含 HTML 看板(纯前端 canvas 图表, 无 CDN,
    # 轮询同源 /metrics); 0/off/false/no = 关(不挂路由)。见
    # entrypoints/serve/instrumentator/monitor.py。
    "VLLM_MONITOR": _monitor,
    # A3(sm75 参考): custom allreduce 在 cuda graph capture 时的图输入策略。
    # auto=full decode 走 registered 快路径, piecewise/prefill 回退 staging
    # buffer(sm75 图私有大 buffer 无法经 CUDA IPC 导出); registered/staging
    # 可强制覆盖。
    "VLLM_CUSTOM_ALLREDUCE_GRAPH_INPUT_MODE": lambda: os.getenv(
        "VLLM_CUSTOM_ALLREDUCE_GRAPH_INPUT_MODE", "auto"
    ),
    # vllm-sm75 overlay: idle auto-sleep. Populated by EngineArgs from the
    # --auto-sleep-* CLI flags; consumed inside the engine-core process by
    # vllm.v1.engine.auto_sleep(那里用 os.environ.get 直读, 此处注册仅为让
    # validate_environ 不告警 + 被 compile_factors 看到后由 INSTALL_IGNORED 排除)。
    # Idle minutes (float) before the engine auto-sleeps; 0 disables.
    "VLLM_AUTO_SLEEP_IDLE_TIMEOUT": lambda: float(
        os.getenv("VLLM_AUTO_SLEEP_IDLE_TIMEOUT", "0")
    ),
    # 'cpu' (sleep level 1, pinned CPU backup) or 'reload' (sleep level 2,
    # weights discarded and reloaded from the checkpoint on wake).
    "VLLM_AUTO_SLEEP_OFFLOAD_TARGET": lambda: os.getenv(
        "VLLM_AUTO_SLEEP_OFFLOAD_TARGET", "cpu"
    ),
    # Checkpoint path used to reload weights on wake for the 'reload' target.
    "VLLM_AUTO_SLEEP_RELOAD_PATH": lambda: os.getenv("VLLM_AUTO_SLEEP_RELOAD_PATH", ""),
    # Seconds between page-cache re-warm ticks while sleeping (reload mode);
    # keeps the checkpoint in the OS page cache so the wake-time reload read
    # is fast. 0 disables the background keeper (one-shot warm on sleep/wake
    # still happens).
    "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL": lambda: float(
        os.getenv("VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL", "600")
    ),
}


def apply() -> None:
    """把 SM75 扩展 env 注入上游 vllm.envs(幂等)。

    由 install_sm75_overlay.py append 到上游 envs.py 尾部的
    `import vllm.envs_sm75; vllm.envs_sm75.apply()` 触发。此刻 envs 模块的
    environment_variables / compile_factors 均已定义(append 在模块体最末)。
    用 sys.modules 取正在加载的 vllm.envs(即 append 所在模块本身), 不触发
    `import vllm` 的副作用链; 测试里 patch sys.modules["vllm.envs"] 即可复用。
    """
    import sys

    envs = sys.modules["vllm.envs"]

    # 1) 注入 getter —— dict.update 天然幂等(同 key 覆盖同值)。
    envs.environment_variables.update(EXTENSIONS)

    # 2) 包 compile_factors, 从 hash factors pop 掉 INSTALL_IGNORED(幂等: 已包跳过)。
    if not getattr(envs.compile_factors, "_sm75_wrapped", False):
        _orig = envs.compile_factors

        def _compile_factors_sm75():
            factors = _orig()
            for key in INSTALL_IGNORED:
                factors.pop(key, None)
            return factors

        _compile_factors_sm75._sm75_wrapped = True  # type: ignore[attr-defined]
        _compile_factors_sm75.__wrapped__ = _orig
        envs.compile_factors = _compile_factors_sm75
