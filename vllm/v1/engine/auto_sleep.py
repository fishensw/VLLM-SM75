# SPDX-License-Identifier: Apache-2.0
# vllm-sm75 overlay: idle auto-sleep controller for the V1 EngineCore.
#
# Upstream vLLM ships --enable-sleep-mode, but sleeping/waking must be driven
# by external POST /sleep + POST /wake_up calls.  This controller adds the
# missing control plane:
#
#   * the engine sleeps by itself after a configurable idle timeout;
#   * the first arriving request wakes it transparently (no API call).
#
# Design notes (upstream v0.28.0, see tmp/PLAN-auto-sleep-mode.md):
#
#   * When fully idle the EngineCoreProc main loop blocks in
#     input_queue.get(block=True) and step() returns early, so a per-step
#     timeout check is impossible.  The controller reuses upstream facilities:
#       - the one-shot _idle_state_callbacks to observe "engine went idle";
#       - a daemon threading.Timer armed once per idle period;
#       - input_queue.put_nowait((WAKEUP, None)) (the same sentinel the
#         shutdown signal handler uses) to hand the decision back to the
#         main loop thread, where sleep()/wake_up() may only run.
#   * offload_target "cpu"    -> sleep(level=1): weights copied to pinned CPU
#                                memory, wake is fast (~1-2 s);
#     offload_target "reload" -> sleep(level=2): weights discarded, wake is
#     wake_up() + collective_rpc("reload_weights") which re-runs the startup
#     load pipeline (quantization repack included), wake is slow (~20-60 s);
#     offload_target "exit"   -> deep sleep: the whole EngineCore process
#                                exits (freeing weights, the CUDA context and
#                                the TP worker processes, GPU power -> P8).
#                                The client keeps serving and respawns the
#                                engine transparently on the next request
#                                (cold start, ~1-3 min, paid by that request's
#                                TTFT).  See tmp/PLAN-deep-sleep-exit.md.
#
#   * Page-cache pre-warming (reload and exit modes): the wake-time
#     ``reload_weights`` read (and the exit-mode respawn's cold-start load)
#     is a 20+ GB disk read.  To keep it fast, the controller hints the OS to
#     retain the checkpoint's .safetensors pages in the page cache
#     (``posix_fadvise(POSIX_FADV_WILLNEED)``).  For reload mode it warms once
#     on sleep, once on wake, and on a background timer while sleeping; for
#     exit mode it warms once right before the process exits (the OS page
#     cache outlives the process, so the respawn reads warm).  This is a pure
#     performance hint: async, non-blocking, no-op for pages already resident.
#
# This module deliberately has no vllm imports so it can be unit-tested
# without a GPU environment; the EngineCore is accessed duck-typed.

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

IDLE_TIMEOUT_ENV = "VLLM_AUTO_SLEEP_IDLE_TIMEOUT"
OFFLOAD_TARGET_ENV = "VLLM_AUTO_SLEEP_OFFLOAD_TARGET"
RELOAD_PATH_ENV = "VLLM_AUTO_SLEEP_RELOAD_PATH"
PAGE_CACHE_KEEP_INTERVAL_ENV = "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL"

_OFFLOAD_TARGETS = ("cpu", "reload", "exit")

# Default interval (seconds) between page-cache re-warm ticks while the
# engine sleeps in reload mode.  0 disables the background keeper (the
# one-shot warm on sleep/wake still happens).
DEFAULT_PAGE_CACHE_KEEP_INTERVAL_SECONDS = 600.0


def _parse_page_cache_keep_interval() -> float:
    """Parse the page-cache keeper interval from the environment.

    An unset or invalid value falls back to the default rather than
    disabling auto-sleep: a bad interval is a tuning mistake, not a reason
    to turn the whole feature off.
    """
    raw = os.environ.get(PAGE_CACHE_KEEP_INTERVAL_ENV, "")
    if not raw:
        return DEFAULT_PAGE_CACHE_KEEP_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "auto-sleep: invalid %s=%r, using default %.0fs",
            PAGE_CACHE_KEEP_INTERVAL_ENV,
            raw,
            DEFAULT_PAGE_CACHE_KEEP_INTERVAL_SECONDS,
        )
        return DEFAULT_PAGE_CACHE_KEEP_INTERVAL_SECONDS
    if value < 0:
        logger.warning(
            "auto-sleep: negative %s=%r, using default %.0fs",
            PAGE_CACHE_KEEP_INTERVAL_ENV,
            value,
            DEFAULT_PAGE_CACHE_KEEP_INTERVAL_SECONDS,
        )
        return DEFAULT_PAGE_CACHE_KEEP_INTERVAL_SECONDS
    return value


@dataclass(frozen=True)
class AutoSleepConfig:
    """Resolved auto-sleep configuration (timeout in seconds)."""

    timeout_seconds: float
    offload_target: str = "cpu"
    reload_path: str = ""
    # Seconds between page-cache re-warm ticks while sleeping (reload mode).
    # Only meaningful for the reload target; 0 disables the background keeper.
    page_cache_keep_interval_seconds: float = DEFAULT_PAGE_CACHE_KEEP_INTERVAL_SECONDS

    @property
    def sleep_level(self) -> int:
        return 1 if self.offload_target == "cpu" else 2

    @property
    def is_exit(self) -> bool:
        """Whether idle should terminate the process (deep sleep)."""
        return self.offload_target == "exit"

    @classmethod
    def from_env(cls) -> "AutoSleepConfig | None":
        """Build config from VLLM_AUTO_SLEEP_* envs; None when disabled.

        The envs are populated by EngineArgs.__post_init__ from the
        --auto-sleep-* CLI flags before the engine-core process is spawned.
        """
        raw = os.environ.get(IDLE_TIMEOUT_ENV, "0")
        try:
            timeout_minutes = float(raw)
        except ValueError:
            logger.warning(
                "auto-sleep: invalid %s=%r, auto-sleep disabled",
                IDLE_TIMEOUT_ENV,
                raw,
            )
            return None
        if timeout_minutes <= 0:
            return None
        target = os.environ.get(OFFLOAD_TARGET_ENV, "cpu")
        if target not in _OFFLOAD_TARGETS:
            raise ValueError(
                f"auto-sleep: {OFFLOAD_TARGET_ENV} must be one of "
                f"{_OFFLOAD_TARGETS}, got {target!r}"
            )
        reload_path = os.environ.get(RELOAD_PATH_ENV, "")
        if target == "reload" and not reload_path:
            raise ValueError(
                f"auto-sleep: offload target 'reload' requires "
                f"{RELOAD_PATH_ENV} set to a readable checkpoint path"
            )
        return cls(
            timeout_seconds=timeout_minutes * 60.0,
            offload_target=target,
            reload_path=reload_path,
            page_cache_keep_interval_seconds=_parse_page_cache_keep_interval(),
        )


class AutoSleepState(str, Enum):
    ACTIVE = "active"
    SLEEPING = "sleeping"
    WAKING = "waking"
    EXITED = "exited"


def warm_safetensors_page_cache(model_path: str) -> int:
    """Hint the OS to retain the checkpoint's .safetensors pages in cache.

    Issues ``posix_fadvise(POSIX_FADV_WILLNEED)`` for every .safetensors
    shard under ``model_path`` (or the file itself, when it is a single
    shard).  The hint is asynchronous and non-blocking, and a no-op for
    pages already resident, so calling it is cheap.  It keeps the wake-time
    ``reload_weights`` read in the page cache instead of on cold NVMe.

    ``model_path`` may be a directory of shards or a single .safetensors
    file.  Returns the number of files successfully advised (0 when there
    is nothing to advise).  Per-file errors are logged and skipped so one
    unreadable shard does not abort the warm.
    """
    if not model_path:
        return 0
    if os.path.isfile(model_path):
        files = [model_path] if model_path.endswith(".safetensors") else []
    elif os.path.isdir(model_path):
        files = glob.glob(os.path.join(model_path, "*.safetensors"))
    else:
        return 0

    advised: list[str] = []
    for path in files:
        try:
            with open(path, "rb") as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
        except OSError:
            logger.warning("page-cache warm: could not advise %s", path)
            continue
        advised.append(path)

    if advised:
        total_bytes = 0
        for path in advised:
            try:
                total_bytes += os.path.getsize(path)
            except OSError:
                pass
        logger.info(
            "page-cache warm: advised %d safetensors file(s) (%.1f GB) under %s",
            len(advised),
            total_bytes / 1e9,
            model_path,
        )
    return len(advised)


class PageCacheKeeper:
    """Background daemon that keeps the checkpoint in the page cache.

    While the engine sleeps in reload mode, each tick re-issues the
    ``POSIX_FADV_WILLNEED`` hint (cheap, no-op for resident pages) so the
    cache is still warm when a request triggers the wake-time reload.  The
    thread is a daemon: it never blocks shutdown and dies with the process.
    """

    def __init__(self, model_path: str, interval_seconds: float) -> None:
        self._model_path = model_path
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="auto-sleep-page-cache-keeper",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        thread, self._thread = self._thread, None
        # The wait is interruptible by the event and each tick is a
        # non-blocking fadvise, so this returns promptly; the bound only
        # guards against a slow tick holding the main thread.
        thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._thread is not None

    def _run(self) -> None:
        # Wait before the first tick: the warm-on-sleep already primed the
        # cache, so the first background refresh happens one interval later.
        while not self._stop_event.wait(self._interval_seconds):
            warm_safetensors_page_cache(self._model_path)


class AutoSleepController:
    """Idle auto-sleep state machine driven by EngineCore hooks.

    All public methods run on the EngineCore main loop thread except the
    timer callback (_poke), which only enqueues the WAKEUP sentinel.
    The engine_core object is accessed duck-typed; required attributes:
    input_queue, scheduler.has_requests(), is_sleeping(), sleep(level),
    wake_up(), model_executor.collective_rpc(), _idle_state_callbacks, and
    optionally is_running() (absent on the in-process EngineCore base class).
    """

    def __init__(
        self,
        engine_core: Any,
        wakeup_request_type: Any,
        config: AutoSleepConfig | None = None,
    ) -> None:
        if config is None:
            config = AutoSleepConfig.from_env()
        self._engine = engine_core
        self._wakeup_request_type = wakeup_request_type
        self._config = config
        self._state = AutoSleepState.ACTIVE
        self._timer: threading.Timer | None = None
        self._keeper: PageCacheKeeper | None = None
        self._last_activity = time.monotonic()
        if config is not None:
            logger.info(
                "auto-sleep enabled: idle_timeout=%.1fs target=%s "
                "reload_path=%s page_cache_keep_interval=%.0fs",
                config.timeout_seconds,
                config.offload_target,
                config.reload_path or "<unset>",
                config.page_cache_keep_interval_seconds,
            )

    @property
    def enabled(self) -> bool:
        return self._config is not None

    @property
    def state(self) -> AutoSleepState:
        return self._state

    # ------------------------------------------------------------------
    # Hooks installed in EngineCore (main loop thread)
    # ------------------------------------------------------------------

    def on_idle(self, engine_core: Any) -> None:
        """Idle-state callback: re-register and arm the sleep timer.

        Upstream pops one-shot callbacks on every idle-loop iteration, so
        re-appending ourselves keeps the controller subscribed for future
        idle periods.
        """
        if not self.enabled:
            return
        engine_core._idle_state_callbacks.append(self.on_idle)
        if self._state is not AutoSleepState.ACTIVE:
            return
        if self._timer is not None:
            return
        if self._engine.is_sleeping() or not self._engine_running():
            return
        delay = max(
            self._config.timeout_seconds - (time.monotonic() - self._last_activity),
            0.0,
        )
        self._timer = threading.Timer(delay, self._poke)
        self._timer.daemon = True
        self._timer.start()
        logger.debug("auto-sleep: armed sleep timer for %.1fs", delay)

    def on_request_arrival(self) -> None:
        """Hook for EngineCore.add_request: refresh idle clock / wake up."""
        if not self.enabled:
            return
        self._last_activity = time.monotonic()
        self._cancel_timer()
        if self._engine.is_sleeping():
            self._wake()
        else:
            # Engine is awake (e.g. woken via the dev endpoint while the
            # controller thought it was asleep): stop any lingering keeper so
            # it is only ever running while actually sleeping.
            self._stop_keeper()

    def on_wakeup_poke(self) -> None:
        """Hook for the EngineCore WAKEUP input branch (main thread).

        The timer only pokes; every decision is re-validated here because
        requests may have arrived in the meantime.
        """
        if not self.enabled or self._state is not AutoSleepState.ACTIVE:
            return
        if self._engine.is_sleeping() or not self._engine_running():
            return
        if self._engine.scheduler.has_requests():
            return
        if time.monotonic() - self._last_activity < self._config.timeout_seconds:
            return
        if self._config.is_exit:
            self._deep_sleep_exit()
        else:
            self._sleep()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _engine_running(self) -> bool:
        is_running = getattr(self._engine, "is_running", None)
        return True if is_running is None else is_running()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _start_keeper(self) -> None:
        """Arm the page-cache keeper (reload mode, interval > 0, path set).

        Primed by an immediate warm, then refreshed by a daemon thread on the
        configured interval so the cache survives long idle periods.
        """
        if self._keeper is not None:
            return
        if self._config.offload_target != "reload":
            return
        if self._config.page_cache_keep_interval_seconds <= 0:
            return
        if not self._config.reload_path:
            return
        warm_safetensors_page_cache(self._config.reload_path)
        self._keeper = PageCacheKeeper(
            self._config.reload_path,
            self._config.page_cache_keep_interval_seconds,
        )
        self._keeper.start()
        logger.info(
            "auto-sleep: page-cache keeper armed (interval=%.0fs, path=%s)",
            self._config.page_cache_keep_interval_seconds,
            self._config.reload_path,
        )

    def _stop_keeper(self) -> None:
        if self._keeper is not None:
            self._keeper.stop()
            self._keeper = None

    def _poke(self) -> None:
        # Timer thread: only nudge the main loop; every decision happens on
        # the main thread in on_wakeup_poke.
        self._timer = None
        self._engine.input_queue.put_nowait((self._wakeup_request_type, None))

    def _set_state(self, state: AutoSleepState) -> None:
        if self._state is not state:
            logger.info(
                "auto-sleep: state %s -> %s", self._state.value, state.value
            )
            self._state = state

    def _sleep(self) -> None:
        self._set_state(AutoSleepState.SLEEPING)
        level = self._config.sleep_level
        try:
            self._engine.sleep(level)
        except Exception:
            logger.exception(
                "auto-sleep: sleep(level=%d) failed; engine stays active",
                level,
            )
            self._set_state(AutoSleepState.ACTIVE)
            return
        self._start_keeper()
        logger.info(
            "auto-sleep: engine is sleeping (level=%d, target=%s)",
            level,
            self._config.offload_target,
        )

    def _deep_sleep_exit(self) -> None:
        """Deep sleep: terminate the engine-core process entirely.

        Frees all GPU memory including the CUDA context and the TP worker
        processes (GPU drops to its deepest idle state).  The client keeps
        serving and respawns the engine transparently on the next request;
        that request's latency includes the cold start.  Runs on the main
        loop thread via on_wakeup_poke.
        """
        request_exit = getattr(self._engine, "request_deep_sleep_exit", None)
        if request_exit is None:
            # The in-process EngineCore base class has no exit path; only the
            # proc-mode EngineCoreProc (vllm serve) supports deep sleep.
            logger.error(
                "auto-sleep: offload target 'exit' requires the engine-core "
                "subprocess (vllm serve); staying active"
            )
            self._set_state(AutoSleepState.ACTIVE)
            return
        self._set_state(AutoSleepState.EXITED)
        # Warm the page cache so the respawn's model load reads from cache;
        # the OS page cache outlives this process.
        if self._config.reload_path:
            warm_safetensors_page_cache(self._config.reload_path)
        request_exit()

    def _wake(self) -> None:
        self._set_state(AutoSleepState.WAKING)
        if self._config.offload_target == "reload":
            # Stop the background keeper and re-hint the cache so the
            # reload_weights read below starts from a warm page cache.  This
            # runs before wake_up() to overlap the kernel's background read
            # with the remap/buffer restore.
            self._stop_keeper()
            if self._config.reload_path:
                warm_safetensors_page_cache(self._config.reload_path)
        start = time.monotonic()
        try:
            self._engine.wake_up()
            if self._config.offload_target == "reload":
                # Re-run the startup load pipeline (incl. quantization
                # repack) on every worker; weights are written back in place.
                self._engine.model_executor.collective_rpc(
                    "reload_weights",
                    kwargs={
                        "weights_path": self._config.reload_path,
                        "is_checkpoint_format": True,
                    },
                )
        except Exception:
            logger.exception("auto-sleep: wake_up failed")
            self._set_state(AutoSleepState.ACTIVE)
            raise
        elapsed = time.monotonic() - start
        self._set_state(AutoSleepState.ACTIVE)
        logger.info(
            "auto-sleep: engine woke in %.1fs (target=%s); the first "
            "request latency includes this time",
            elapsed,
            self._config.offload_target,
        )
