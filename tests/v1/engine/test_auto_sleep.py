# SPDX-License-Identifier: Apache-2.0
# vllm-sm75 overlay tests: auto-sleep config + controller state machine.
#
# Runs two ways:
#   * in the SM75 image under pytest (vllm importable, overlay installed);
#   * locally with plain ``python3 tests/v1/engine/test_auto_sleep.py``
#     (no vllm/torch): the module under test is loaded from its file path.

from __future__ import annotations

import importlib.util
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _load_auto_sleep() -> Any:
    try:
        from vllm.v1.engine import auto_sleep
    except ImportError:
        # Local run: the repo mirrors only overlay files (no package inits),
        # so load the module straight from its file path.
        path = (
            Path(__file__).resolve().parents[3]
            / "vllm"
            / "v1"
            / "engine"
            / "auto_sleep.py"
        )
        spec = importlib.util.spec_from_file_location("auto_sleep_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        # Register before exec: @dataclass resolves string annotations via
        # sys.modules[cls.__module__].
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return auto_sleep


auto_sleep = _load_auto_sleep()


@contextmanager
def _env(pairs: dict[str, str | None]):
    saved = {key: os.environ.get(key) for key in pairs}
    for key, value in pairs.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _FakeScheduler:
    def __init__(self) -> None:
        self.has_requests_flag = False

    def has_requests(self) -> bool:
        return self.has_requests_flag


class _FakeExecutor:
    def __init__(self, fake: "_FakeEngine") -> None:
        self._fake = fake

    def collective_rpc(self, method: str, kwargs: dict | None = None):
        self._fake.rpc_calls.append((method, kwargs))
        return None


class _FakeEngine:
    """Duck-typed stand-in for EngineCore (attributes used by the controller)."""

    def __init__(self) -> None:
        self.input_queue: queue.Queue = queue.Queue()
        self.scheduler = _FakeScheduler()
        self.model_executor = _FakeExecutor(self)
        self._idle_state_callbacks: list = []
        self.sleeping = False
        self.running = True
        self.sleep_calls: list[int] = []
        self.wake_up_calls = 0
        self.rpc_calls: list[tuple[str, dict | None]] = []
        self.wake_up_error: Exception | None = None
        # Count of request_deep_sleep_exit() calls.  The method itself only
        # exists on EngineCoreProc; tests attach it via _enable_deep_sleep_exit
        # so the controller's getattr fallback can be exercised too.
        self.deep_sleep_exit_calls = 0

    def is_sleeping(self) -> bool:
        return self.sleeping

    def is_running(self) -> bool:
        return self.running

    def sleep(self, level: int = 1, mode: str = "abort") -> None:
        self.sleep_calls.append(level)
        self.sleeping = True

    def wake_up(self, tags: list[str] | None = None) -> None:
        if self.wake_up_error is not None:
            raise self.wake_up_error
        self.sleeping = False
        self.wake_up_calls += 1


def _make_controller(
    fake: _FakeEngine,
    auto_sleep_mod: Any,
    timeout_seconds: float = 0.05,
    offload_target: str = "cpu",
    reload_path: str = "",
    page_cache_keep_interval_seconds: float = 600.0,
) -> Any:
    config = auto_sleep_mod.AutoSleepConfig(
        timeout_seconds=timeout_seconds,
        offload_target=offload_target,
        reload_path=reload_path,
        page_cache_keep_interval_seconds=page_cache_keep_interval_seconds,
    )
    return auto_sleep_mod.AutoSleepController(fake, b"\x05", config=config)


def _drain_wakeup(fake: _FakeEngine, controller: Any, timeout: float = 3.0) -> None:
    """Wait for the timer to enqueue the WAKEUP sentinel, then dispatch it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            request_type, _ = fake.input_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.005)
            continue
        if request_type == controller._wakeup_request_type:
            controller.on_wakeup_poke()
            return
    raise AssertionError("WAKEUP sentinel was not enqueued by the timer")


def _enable_deep_sleep_exit(fake: _FakeEngine) -> None:
    """Attach request_deep_sleep_exit (present only on EngineCoreProc)."""

    def _request_deep_sleep_exit() -> None:
        fake.deep_sleep_exit_calls += 1

    fake.request_deep_sleep_exit = _request_deep_sleep_exit


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_config_from_env_disabled_without_envs():
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: None,
            auto_sleep.OFFLOAD_TARGET_ENV: None,
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        assert auto_sleep.AutoSleepConfig.from_env() is None


def test_config_from_env_parses_minutes_and_maps_target():
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "0.2",  # 0.2 minutes = 12 seconds
            auto_sleep.OFFLOAD_TARGET_ENV: "cpu",
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        config = auto_sleep.AutoSleepConfig.from_env()
    assert config is not None
    assert config.timeout_seconds == 12.0
    assert config.offload_target == "cpu"
    assert config.sleep_level == 1


def test_config_from_env_reload_requires_path():
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "1",
            auto_sleep.OFFLOAD_TARGET_ENV: "reload",
            auto_sleep.RELOAD_PATH_ENV: "/ckpt",
        }
    ):
        config = auto_sleep.AutoSleepConfig.from_env()
    assert config is not None
    assert config.sleep_level == 2
    assert config.reload_path == "/ckpt"

    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "1",
            auto_sleep.OFFLOAD_TARGET_ENV: "reload",
            auto_sleep.RELOAD_PATH_ENV: "",
        }
    ):
        try:
            auto_sleep.AutoSleepConfig.from_env()
        except ValueError as exc:
            assert "RELOAD_PATH" in str(exc)
        else:
            raise AssertionError("expected ValueError for reload without path")


def test_config_from_env_invalid_values():
    with _env({auto_sleep.IDLE_TIMEOUT_ENV: "not-a-number"}):
        assert auto_sleep.AutoSleepConfig.from_env() is None
    with _env({auto_sleep.IDLE_TIMEOUT_ENV: "-1"}):
        assert auto_sleep.AutoSleepConfig.from_env() is None
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "1",
            auto_sleep.OFFLOAD_TARGET_ENV: "gpu",
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        try:
            auto_sleep.AutoSleepConfig.from_env()
        except ValueError as exc:
            assert "OFFLOAD_TARGET" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown target")


def test_config_from_env_exit_target():
    # "exit" deep sleep does not require a reload path (it re-runs the full
    # startup load from the model path on respawn).
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "1",
            auto_sleep.OFFLOAD_TARGET_ENV: "exit",
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        config = auto_sleep.AutoSleepConfig.from_env()
    assert config is not None
    assert config.offload_target == "exit"
    assert config.is_exit


# ---------------------------------------------------------------------------
# Controller state machine
# ---------------------------------------------------------------------------


def test_disabled_controller_hooks_are_noops():
    fake = _FakeEngine()
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: None,
            auto_sleep.OFFLOAD_TARGET_ENV: None,
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        controller = auto_sleep.AutoSleepController(fake, b"\x05")
        assert not controller.enabled
    fake._idle_state_callbacks = [controller.on_idle]
    controller.on_idle(fake)  # must not re-register or touch the engine
    assert fake._idle_state_callbacks == [controller.on_idle]
    controller.on_request_arrival()
    controller.on_wakeup_poke()
    assert fake.sleep_calls == []
    assert fake.wake_up_calls == 0


def test_idle_arms_timer_and_wakeup_poke_sleeps():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    assert controller.enabled

    controller.on_idle(fake)
    assert fake._idle_state_callbacks == [controller.on_idle], "callback must re-register"
    assert controller._timer is not None, "timer must be armed while idle"

    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [1], "cpu target must sleep at level 1"
    assert controller.state is auto_sleep.AutoSleepState.SLEEPING
    assert fake.sleeping


def test_notify_idle_callbacks_terminates_with_reregistering_callback():
    """Regression: a callback that re-registers itself during notification
    must not spin EngineCore._notify_idle_state_callbacks forever.

    AutoSleepController.on_idle re-appends itself to
    engine._idle_state_callbacks on every invocation (to stay subscribed
    across idle periods).  The previous implementation popped the live list
    inside a ``while self._idle_state_callbacks:`` loop, so the
    re-registration kept the list non-empty and the loop never terminated:
    the engine sat idle with one CPU core pegged and the API server never
    became ready.  The snapshot-based implementation must return after one
    pass and defer the re-registration to the next idle notification.

    Note: the method lives on EngineCoreProc (the process engine the API
    server runs), not on the in-process EngineCore base class, which only
    declares the _idle_state_callbacks attribute.
    """
    try:
        from vllm.v1.engine.core import EngineCoreProc
    except ImportError:
        print("  skipped: vllm not importable in this environment")
        return

    notify = EngineCoreProc._notify_idle_state_callbacks  # plain function

    class _Stub:
        def __init__(self) -> None:
            self._idle_state_callbacks: list = []

    stub = _Stub()
    calls = 0

    def reregistering(core) -> None:
        nonlocal calls
        calls += 1
        core._idle_state_callbacks.append(reregistering)

    stub._idle_state_callbacks.append(reregistering)

    # Run in a worker thread so a regression (infinite loop) fails fast
    # instead of hanging the whole test run.
    done = []
    worker = threading.Thread(
        target=lambda: (notify(stub), done.append(1)), daemon=True
    )
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), (
        "_notify_idle_state_callbacks hung on a re-registering callback"
    )
    assert done, "idle notification must complete"
    assert calls == 1, "callback must fire exactly once per notification"
    assert stub._idle_state_callbacks == [reregistering], (
        "re-registration must be deferred to the next idle notification"
    )


def test_reload_target_sleeps_at_level_2():
    fake = _FakeEngine()
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="reload",
        reload_path="/ckpt",
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [2]


def test_request_arrival_cancels_armed_timer():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    controller.on_idle(fake)
    assert controller._timer is not None

    controller.on_request_arrival()
    assert controller._timer is None

    time.sleep(0.1)  # past the timeout: the cancelled timer must not fire
    assert fake.input_queue.empty()
    assert fake.sleep_calls == []


def test_wakeup_poke_ignores_pending_requests():
    fake = _FakeEngine()
    fake.scheduler.has_requests_flag = True
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.001)
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == []
    assert controller.state is auto_sleep.AutoSleepState.ACTIVE


def test_wakeup_poke_ignores_fresh_activity():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=60.0)
    controller.on_request_arrival()  # fresh last_activity
    # Poke directly (as the timer would) without waiting.
    controller.on_wakeup_poke()
    assert fake.sleep_calls == []


def test_request_wakes_sleeping_engine_and_reloads():
    fake = _FakeEngine()
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="reload",
        reload_path="/ckpt",
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [2]

    controller.on_request_arrival()
    assert fake.sleeping is False, "wake_up must resume the engine"
    assert fake.wake_up_calls == 1
    assert fake.rpc_calls == [
        ("reload_weights", {"weights_path": "/ckpt", "is_checkpoint_format": True})
    ], "reload target must re-run the checkpoint load pipeline"
    assert controller.state is auto_sleep.AutoSleepState.ACTIVE


def test_cpu_target_wake_does_not_reload():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [1]

    controller.on_request_arrival()
    assert fake.sleeping is False
    assert fake.rpc_calls == [], "cpu target must not reload weights"


def test_wake_up_failure_propagates_and_stays_active():
    fake = _FakeEngine()
    fake.wake_up_error = RuntimeError("boom")
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    fake.sleeping = True  # engine went to sleep via the dev endpoint

    try:
        controller.on_request_arrival()
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("wake_up failure must propagate to the caller")
    assert controller.state is auto_sleep.AutoSleepState.ACTIVE


def test_no_double_sleep_while_sleeping():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    fake.sleeping = True  # manual sleep via dev endpoint, controller unaware

    controller.on_idle(fake)
    assert controller._timer is None, "no timer while engine is already sleeping"

    controller.on_wakeup_poke()
    assert fake.sleep_calls == [], "must not sleep an already-sleeping engine"


def test_no_sleep_during_shutdown():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.001)
    controller.on_idle(fake)
    fake.running = False
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == []


# ---------------------------------------------------------------------------
# Page-cache warming
# ---------------------------------------------------------------------------


def test_warm_page_cache_nonexistent_path():
    assert auto_sleep.warm_safetensors_page_cache("/nonexistent/model") == 0
    assert auto_sleep.warm_safetensors_page_cache("") == 0


def test_warm_page_cache_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        assert auto_sleep.warm_safetensors_page_cache(tmp) == 0


def test_warm_page_cache_counts_safetensors_only():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "model-00001.safetensors").write_bytes(b"x" * 100)
        (Path(tmp) / "model-00002.safetensors").write_bytes(b"y" * 200)
        (Path(tmp) / "config.json").write_text("{}")
        (Path(tmp) / "weights.bin").write_bytes(b"z" * 50)
        # nested shards are not scanned (top-level glob only)
        nested = Path(tmp) / "shards"
        nested.mkdir()
        (nested / "model-00003.safetensors").write_bytes(b"w" * 10)
        assert auto_sleep.warm_safetensors_page_cache(tmp) == 2


def test_warm_page_cache_single_file():
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        shard.write_bytes(b"x" * 100)
        assert auto_sleep.warm_safetensors_page_cache(str(shard)) == 1
        other = Path(tmp) / "weights.bin"
        other.write_bytes(b"z" * 100)
        assert auto_sleep.warm_safetensors_page_cache(str(other)) == 0


def test_config_page_cache_interval_default():
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "1",
            auto_sleep.OFFLOAD_TARGET_ENV: "reload",
            auto_sleep.RELOAD_PATH_ENV: "/ckpt",
            auto_sleep.PAGE_CACHE_KEEP_INTERVAL_ENV: None,
        }
    ):
        config = auto_sleep.AutoSleepConfig.from_env()
    assert config is not None
    assert config.page_cache_keep_interval_seconds == 600.0


def test_config_page_cache_interval_explicit():
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "1",
            auto_sleep.OFFLOAD_TARGET_ENV: "reload",
            auto_sleep.RELOAD_PATH_ENV: "/ckpt",
            auto_sleep.PAGE_CACHE_KEEP_INTERVAL_ENV: "30",
        }
    ):
        config = auto_sleep.AutoSleepConfig.from_env()
    assert config is not None
    assert config.page_cache_keep_interval_seconds == 30.0


def test_config_page_cache_interval_invalid_falls_back_to_default():
    for bad in ("not-a-number", "-5"):
        with _env(
            {
                auto_sleep.IDLE_TIMEOUT_ENV: "1",
                auto_sleep.OFFLOAD_TARGET_ENV: "reload",
                auto_sleep.RELOAD_PATH_ENV: "/ckpt",
                auto_sleep.PAGE_CACHE_KEEP_INTERVAL_ENV: bad,
            }
        ):
            config = auto_sleep.AutoSleepConfig.from_env()
        assert config is not None, "bad interval must not disable auto-sleep"
        assert config.page_cache_keep_interval_seconds == 600.0


def test_reload_sleep_arms_page_cache_keeper():
    fake = _FakeEngine()
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="reload",
        reload_path="/ckpt",
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [2]
    try:
        assert controller._keeper is not None, "reload sleep must arm the keeper"
        assert controller._keeper.is_running()
    finally:
        controller._stop_keeper()
    assert controller._keeper is None


def test_cpu_sleep_does_not_arm_keeper():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [1]
    assert controller._keeper is None, "cpu target keeps no page-cache keeper"


def test_reload_interval_zero_disables_keeper():
    fake = _FakeEngine()
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="reload",
        reload_path="/ckpt", page_cache_keep_interval_seconds=0.0,
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [2]
    assert controller._keeper is None, "interval 0 disables the background keeper"


def test_wake_stops_page_cache_keeper():
    fake = _FakeEngine()
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="reload",
        reload_path="/ckpt",
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert controller._keeper is not None
    controller.on_request_arrival()  # triggers _wake()
    assert fake.sleeping is False
    assert controller._keeper is None, "wake must stop the keeper"
    assert controller.state is auto_sleep.AutoSleepState.ACTIVE


def test_request_on_awake_engine_stops_lingering_keeper():
    fake = _FakeEngine()
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="reload",
        reload_path="/ckpt",
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert controller._keeper is not None
    # Simulate a manual wake via the dev endpoint: the engine is awake but
    # the controller (and its keeper) are unaware of it.
    fake.sleeping = False
    controller.on_request_arrival()
    assert controller._keeper is None, (
        "a request on an already-awake engine must stop the lingering keeper"
    )


def test_exit_mode_requests_deep_sleep_exit():
    fake = _FakeEngine()
    _enable_deep_sleep_exit(fake)
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="exit",
        reload_path="/ckpt",
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [], "exit mode must not call engine.sleep()"
    assert fake.deep_sleep_exit_calls == 1
    assert controller.state is auto_sleep.AutoSleepState.EXITED


def test_exit_mode_unsupported_engine_stays_active():
    # No request_deep_sleep_exit attached: the in-process EngineCore base
    # class has no exit path, so the controller must stay active (and log an
    # error) rather than wedging the engine.
    fake = _FakeEngine()
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="exit",
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.deep_sleep_exit_calls == 0
    assert controller.state is auto_sleep.AutoSleepState.ACTIVE


# ---------------------------------------------------------------------------
# CLI arg validation (needs the full vllm package; image-only)
# ---------------------------------------------------------------------------


def test_cli_arg_validation_and_env_propagation():
    try:
        from vllm.engine import arg_utils
    except ImportError:
        print("  skipped: vllm not importable in this environment")
        return

    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: None,
            auto_sleep.OFFLOAD_TARGET_ENV: None,
            auto_sleep.RELOAD_PATH_ENV: None,
            auto_sleep.PAGE_CACHE_KEEP_INTERVAL_ENV: None,
        }
    ):
        # timeout without --enable-sleep-mode must be rejected
        try:
            arg_utils.EngineArgs(
                model="/m", auto_sleep_idle_timeout=1.0, enable_sleep_mode=False
            )
        except ValueError as exc:
            assert "enable-sleep-mode" in str(exc)
        else:
            raise AssertionError("expected ValueError without --enable-sleep-mode")

        # negative page-cache keep interval must be rejected
        try:
            arg_utils.EngineArgs(
                model="/m",
                auto_sleep_idle_timeout=1.0,
                auto_sleep_page_cache_keep_interval=-1.0,
                enable_sleep_mode=True,
            )
        except ValueError as exc:
            assert "page-cache-keep-interval" in str(exc)
        else:
            raise AssertionError("expected ValueError for negative keep interval")

        # explicit values propagate to the envs for the engine-core subprocess
        arg_utils.EngineArgs(
            model="/m",
            auto_sleep_idle_timeout=2.5,
            auto_sleep_offload_target="reload",
            auto_sleep_page_cache_keep_interval=30.0,
            enable_sleep_mode=True,
        )
        assert os.environ[auto_sleep.IDLE_TIMEOUT_ENV] == "2.5"
        assert os.environ[auto_sleep.OFFLOAD_TARGET_ENV] == "reload"
        assert os.environ[auto_sleep.RELOAD_PATH_ENV] == "/m"
        assert os.environ[auto_sleep.PAGE_CACHE_KEEP_INTERVAL_ENV] == "30.0"

        # explicit reload path wins over the model path
        arg_utils.EngineArgs(
            model="/m",
            auto_sleep_idle_timeout=2.5,
            auto_sleep_offload_target="reload",
            auto_sleep_reload_path="/other",
            enable_sleep_mode=True,
        )
        assert os.environ[auto_sleep.RELOAD_PATH_ENV] == "/other"

        # disabled by default: no env side effects.  Clear what the cases
        # above set, then verify a fresh EngineArgs (timeout=0) adds nothing
        # back — otherwise the "not in os.environ" check is contaminated by
        # the earlier propagation assertions.
        for key in (
            auto_sleep.IDLE_TIMEOUT_ENV,
            auto_sleep.OFFLOAD_TARGET_ENV,
            auto_sleep.RELOAD_PATH_ENV,
            auto_sleep.PAGE_CACHE_KEEP_INTERVAL_ENV,
        ):
            os.environ.pop(key, None)
        arg_utils.EngineArgs(model="/m", enable_sleep_mode=True)
        for key in (
            auto_sleep.IDLE_TIMEOUT_ENV,
            auto_sleep.OFFLOAD_TARGET_ENV,
            auto_sleep.RELOAD_PATH_ENV,
            auto_sleep.PAGE_CACHE_KEEP_INTERVAL_ENV,
        ):
            assert key not in os.environ


if __name__ == "__main__":
    failures = 0
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
