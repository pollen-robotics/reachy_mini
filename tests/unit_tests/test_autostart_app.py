import asyncio
import types
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from threading import Event

import pytest
from fastapi import HTTPException

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.daemon import startup_app_config
from reachy_mini.daemon.app.routers import apps as apps_router
from reachy_mini.daemon.app.startup_app import (
    AntennaTouchDetector,
    _antennas_in_commanded_motion,
    ensure_startup_app_installed,
    make_startup_app_launcher,
    play_awake_startup_cue,
    rearm_startup_app_watcher,
    start_startup_app,
    start_startup_app_if_idle,
    wake_or_start_startup_app_if_idle,
    watch_antennas_for_startup_app,
)
from reachy_mini.daemon.robot_app_lock import RobotAppLock
from reachy_mini.io.protocol import MotorControlMode


class StubAppManager:
    """Records install/start calls and serves canned app lists per source."""

    def __init__(self, installed: list[str], catalog: list[str]) -> None:
        self._installed = installed
        self._catalog = catalog
        self.installed_calls: list[str] = []
        self.start_attempts: list[str] = []
        self.started: list[str] = []
        self.evict_remote_values: list[bool] = []
        self.running = False

    async def list_available_apps(self, source: SourceKind) -> list[AppInfo]:
        names = self._installed if source == SourceKind.INSTALLED else self._catalog
        return [AppInfo(name=n, source_kind=source) for n in names]

    async def install_new_app(self, app: AppInfo, logger: object) -> None:
        self.installed_calls.append(app.name)
        self._installed.append(app.name)

    async def remove_app(self, name: str, logger: object) -> None:
        if name in self._installed:
            self._installed.remove(name)

    def is_app_running(self) -> bool:
        return self.running

    async def start_app(self, name: str, *, evict_remote: bool = True) -> None:
        self.start_attempts.append(name)
        if self.running:
            raise RuntimeError("An app is already running")
        self.started.append(name)
        self.evict_remote_values.append(evict_remote)
        self.running = True


class StubBackend:
    """Mutable backend surface used by the antenna watcher tests."""

    def __init__(self) -> None:
        self.ready = Event()
        self.ready.set()
        self.present = [0.0, 0.0]
        self.target_antenna_joint_positions: list[float] | None = [0.0, 0.0]
        self.motor_control_mode = MotorControlMode.Enabled
        self.wake_up_calls = 0
        self.played_sounds: list[str] = []
        self.goto_target_calls = 0
        self.on_wake_up_callback: Callable[[], None] | None = None

    def get_present_antenna_joint_positions(self) -> list[float]:
        return self.present

    def get_motor_control_mode(self) -> MotorControlMode:
        return self.motor_control_mode

    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        self.motor_control_mode = mode

    def set_on_wake_up_callback(self, callback: Callable[[], None]) -> None:
        self.on_wake_up_callback = callback

    async def wake_up(self) -> None:
        self.wake_up_calls += 1
        if self.on_wake_up_callback is not None:
            self.on_wake_up_callback()

    def play_sound(self, sound_file: str) -> None:
        self.played_sounds.append(sound_file)

    async def goto_target(self, *args: object, **kwargs: object) -> None:
        self.goto_target_calls += 1


class StubDaemon:
    """Daemon-like object with just the fields the watcher needs."""

    def __init__(self) -> None:
        self.backend = StubBackend()
        self.robot_app_lock = RobotAppLock()


@pytest.mark.asyncio
async def test_already_installed_is_ready_without_install() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=["foo", "bar"])
    assert await ensure_startup_app_installed(mgr, "foo") is True  # type: ignore[arg-type]
    assert mgr.installed_calls == []


@pytest.mark.asyncio
async def test_missing_app_installed_from_catalog() -> None:
    mgr = StubAppManager(installed=[], catalog=["bar"])
    assert await ensure_startup_app_installed(mgr, "bar") is True  # type: ignore[arg-type]
    assert mgr.installed_calls == ["bar"]


@pytest.mark.asyncio
async def test_unknown_app_not_ready_and_not_installed() -> None:
    mgr = StubAppManager(installed=[], catalog=["bar"])
    assert await ensure_startup_app_installed(mgr, "nope") is False  # type: ignore[arg-type]
    assert mgr.installed_calls == []


@pytest.mark.asyncio
async def test_install_failure_returns_false() -> None:
    mgr = StubAppManager(installed=[], catalog=["bar"])

    async def boom(app: AppInfo, logger: object) -> None:
        raise RuntimeError("network down")

    mgr.install_new_app = boom  # type: ignore[assignment]
    assert await ensure_startup_app_installed(mgr, "bar") is False  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_start_calls_start_app() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    await start_startup_app(mgr, "foo")  # type: ignore[arg-type]
    assert mgr.started == ["foo"]
    assert mgr.evict_remote_values == [False]


def test_antenna_touch_detector_triggers_once_until_release() -> None:
    detector = AntennaTouchDetector(press_delta_rad=0.25, release_delta_rad=0.1)

    assert detector.update((0.0, 0.0)) is False
    assert detector.update((0.20, 0.0)) is False
    assert detector.update((0.26, 0.0)) is True
    assert detector.update((0.30, 0.0)) is False
    assert detector.update((0.05, 0.0)) is False
    assert detector.update((0.0, -0.26)) is True


def test_antenna_touch_detector_uses_present_baseline() -> None:
    detector = AntennaTouchDetector(press_delta_rad=0.25, release_delta_rad=0.1)

    assert detector.update((1.0, -1.0)) is False
    assert detector.update((1.2, -1.0)) is False
    assert detector.update((1.3, -1.0)) is True
    assert detector.update((1.3, -1.0)) is False
    assert detector.update((1.05, -1.0)) is False
    assert detector.update((1.0, -1.3)) is True


def test_antenna_touch_detector_arms_when_idle_pose_is_offset() -> None:
    detector = AntennaTouchDetector(press_delta_rad=0.25, release_delta_rad=0.1)

    assert detector.update((0.18, 0.0)) is False
    assert detector.update((0.40, 0.0)) is False
    assert detector.update((0.44, 0.0)) is True


@pytest.mark.asyncio
async def test_antenna_touch_start_uses_non_evicting_app_start() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()

    assert await start_startup_app_if_idle(mgr, daemon, "foo") is True  # type: ignore[arg-type]

    assert mgr.started == ["foo"]
    assert mgr.evict_remote_values == [False]


@pytest.mark.asyncio
async def test_antenna_touch_start_skips_when_remote_session_is_active() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()
    assert daemon.robot_app_lock.try_acquire_remote("remote") is True

    assert await start_startup_app_if_idle(mgr, daemon, "foo") is False  # type: ignore[arg-type]

    assert mgr.started == []


@pytest.mark.asyncio
async def test_antenna_touch_wakes_sleeping_robot_before_starting_app() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()
    daemon.backend.motor_control_mode = MotorControlMode.Disabled

    assert await wake_or_start_startup_app_if_idle(mgr, daemon, "foo") is True  # type: ignore[arg-type]

    assert daemon.backend.motor_control_mode == MotorControlMode.Enabled
    assert daemon.backend.wake_up_calls == 1
    assert daemon.backend.played_sounds == []
    assert daemon.backend.goto_target_calls == 0
    assert mgr.started == ["foo"]
    assert mgr.evict_remote_values == [False]


@pytest.mark.asyncio
async def test_antenna_touch_wake_does_not_duplicate_startup_callback() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()
    daemon.backend.motor_control_mode = MotorControlMode.Disabled
    daemon.backend.on_wake_up_callback = make_startup_app_launcher(
        mgr,
        "foo",  # type: ignore[arg-type]
    )

    assert await wake_or_start_startup_app_if_idle(mgr, daemon, "foo") is True  # type: ignore[arg-type]

    assert daemon.backend.wake_up_calls == 1
    assert mgr.start_attempts == ["foo"]
    assert mgr.started == ["foo"]
    assert mgr.evict_remote_values == [False]


@pytest.mark.asyncio
async def test_antenna_touch_plays_awake_sound_before_starting_app() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()

    assert await wake_or_start_startup_app_if_idle(mgr, daemon, "foo") is True  # type: ignore[arg-type]

    assert daemon.backend.wake_up_calls == 0
    assert daemon.backend.played_sounds == ["wake_up.wav"]
    assert daemon.backend.goto_target_calls == 0
    assert mgr.started == ["foo"]
    assert mgr.evict_remote_values == [False]


@pytest.mark.asyncio
async def test_awake_startup_cue_only_plays_sound() -> None:
    backend = StubBackend()

    await play_awake_startup_cue(backend)

    assert backend.played_sounds == ["wake_up.wav"]
    assert backend.goto_target_calls == 0


@pytest.mark.asyncio
async def test_antenna_watcher_starts_app_on_touch() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()
    task = asyncio.create_task(
        watch_antennas_for_startup_app(
            mgr,  # type: ignore[arg-type]
            daemon,  # type: ignore[arg-type]
            "foo",
            idle_poll_interval_s=0.01,
            blocked_poll_interval_s=0.01,
        )
    )

    try:
        await asyncio.sleep(0.03)
        daemon.backend.present = [0.30, 0.0]
        await asyncio.sleep(0.03)

        assert mgr.started == ["foo"]
        assert mgr.evict_remote_values == [False]
        assert daemon.backend.played_sounds == ["wake_up.wav"]
        assert daemon.backend.goto_target_calls == 0
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_antenna_watcher_resets_detector_after_start_failure() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()
    detector = AntennaTouchDetector()

    async def boom(name: str, *, evict_remote: bool = True) -> None:
        mgr.start_attempts.append(name)
        raise RuntimeError("start failed")

    mgr.start_app = boom  # type: ignore[assignment]
    task = asyncio.create_task(
        watch_antennas_for_startup_app(
            mgr,  # type: ignore[arg-type]
            daemon,  # type: ignore[arg-type]
            "foo",
            detector=detector,
            idle_poll_interval_s=0.01,
            blocked_poll_interval_s=1.0,
        )
    )

    try:
        await asyncio.sleep(0.03)
        daemon.backend.present = [0.30, 0.0]
        for _ in range(20):
            if mgr.start_attempts:
                break
            await asyncio.sleep(0.01)

        assert mgr.start_attempts == ["foo"]
        assert detector._reference is None
        assert detector._armed is False
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_antenna_watcher_wakes_from_baseline_when_target_is_missing() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()
    daemon.backend.motor_control_mode = MotorControlMode.Disabled
    daemon.backend.target_antenna_joint_positions = None
    task = asyncio.create_task(
        watch_antennas_for_startup_app(
            mgr,  # type: ignore[arg-type]
            daemon,  # type: ignore[arg-type]
            "foo",
            idle_poll_interval_s=0.01,
            blocked_poll_interval_s=0.01,
        )
    )

    try:
        await asyncio.sleep(0.03)
        daemon.backend.present = [0.30, 0.0]
        await asyncio.sleep(0.03)

        assert daemon.backend.wake_up_calls == 1
        assert mgr.started == ["foo"]
        assert mgr.evict_remote_values == [False]
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_antenna_watcher_does_not_start_while_app_is_running() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    mgr.running = True
    daemon = StubDaemon()
    daemon.backend.present = [0.30, 0.0]
    task = asyncio.create_task(
        watch_antennas_for_startup_app(
            mgr,  # type: ignore[arg-type]
            daemon,  # type: ignore[arg-type]
            "foo",
            idle_poll_interval_s=0.01,
            blocked_poll_interval_s=0.01,
        )
    )

    try:
        await asyncio.sleep(0.05)
        assert mgr.started == []
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_start_failure_is_swallowed() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])

    async def boom(name: str, *, evict_remote: bool = True) -> None:
        raise RuntimeError("an app is already running")

    mgr.start_app = boom  # type: ignore[assignment]
    await start_startup_app(mgr, "foo")  # must not raise  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_launcher_starts_app_once_across_multiple_wakes() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    launch = make_startup_app_launcher(mgr, "foo")  # type: ignore[arg-type]

    # Simulate several wake-ups; only the first should launch the app.
    launch()
    launch()
    launch()
    await asyncio.sleep(0)  # let the scheduled task(s) run

    assert mgr.started == ["foo"]


@pytest.fixture
def config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the startup-app config at a temp file."""
    path = tmp_path / "daemon_config.json"
    monkeypatch.setattr(startup_app_config, "_config_path", lambda: path)
    return path


def test_config_round_trip(config_file: Path) -> None:
    assert startup_app_config.get_startup_app() is None  # missing file
    startup_app_config.set_startup_app("foo")
    assert startup_app_config.get_startup_app() == "foo"
    startup_app_config.set_startup_app("bar")
    assert startup_app_config.get_startup_app() == "bar"


def test_config_clear(config_file: Path) -> None:
    startup_app_config.set_startup_app("foo")
    startup_app_config.set_startup_app(None)
    assert startup_app_config.get_startup_app() is None


def test_config_corrupt_file_reads_as_none(config_file: Path) -> None:
    config_file.write_text("{ not json")
    assert startup_app_config.get_startup_app() is None


def _stub_request(daemon: object) -> object:
    """Minimal Request stand-in exposing app.state.{daemon,watcher task}."""
    state = types.SimpleNamespace(
        daemon=daemon, startup_app_antenna_watcher_task=None
    )
    return types.SimpleNamespace(app=types.SimpleNamespace(state=state))


@pytest.mark.asyncio
async def test_set_startup_app_endpoint_rejects_uninstalled(config_file: Path) -> None:
    mgr = StubAppManager(installed=["foo"], catalog=["bar"])
    request = _stub_request(StubDaemon())
    with pytest.raises(HTTPException) as exc:
        await apps_router.set_startup_app(
            apps_router.StartupApp(startup_app="bar"), request, mgr  # type: ignore[arg-type]
        )
    assert exc.value.status_code == 400
    assert startup_app_config.get_startup_app() is None  # nothing persisted


@pytest.mark.asyncio
async def test_set_startup_app_endpoint_persists_and_rearms(config_file: Path) -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    request = _stub_request(StubDaemon())
    result = await apps_router.set_startup_app(
        apps_router.StartupApp(startup_app="foo"), request, mgr  # type: ignore[arg-type]
    )
    task = request.app.state.startup_app_antenna_watcher_task  # type: ignore[attr-defined]
    try:
        assert result.startup_app == "foo"
        assert startup_app_config.get_startup_app() == "foo"
        assert (await apps_router.get_startup_app()).startup_app == "foo"
        # Applied live: a watcher was armed without a restart.
        assert task is not None
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_set_startup_app_endpoint_clears_with_null(config_file: Path) -> None:
    startup_app_config.set_startup_app("foo")
    mgr = StubAppManager(installed=["foo"], catalog=[])
    request = _stub_request(StubDaemon())
    result = await apps_router.set_startup_app(
        apps_router.StartupApp(startup_app=None), request, mgr  # type: ignore[arg-type]
    )
    assert result.startup_app is None
    assert startup_app_config.get_startup_app() is None
    assert request.app.state.startup_app_antenna_watcher_task is None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_rearm_arms_watcher_and_wake_hook_for_installed_app() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()

    task = await rearm_startup_app_watcher(mgr, daemon, "foo", None)  # type: ignore[arg-type]
    try:
        assert task is not None
        # The one-shot wake launcher is re-set and starts the new app on wake.
        assert daemon.backend.on_wake_up_callback is not None
        await daemon.backend.wake_up()
        await asyncio.sleep(0)
        assert mgr.started == ["foo"]
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_rearm_cancels_previous_watcher() -> None:
    mgr = StubAppManager(installed=["foo", "bar"], catalog=[])
    daemon = StubDaemon()

    first = await rearm_startup_app_watcher(mgr, daemon, "foo", None)  # type: ignore[arg-type]
    second = await rearm_startup_app_watcher(mgr, daemon, "bar", first)  # type: ignore[arg-type]
    try:
        assert first is not None and first.cancelled()
        assert second is not None and not second.done()
    finally:
        if second is not None:
            second.cancel()
            with suppress(asyncio.CancelledError):
                await second


@pytest.mark.asyncio
async def test_rearm_with_none_cancels_and_neutralizes_wake_hook() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()

    first = await rearm_startup_app_watcher(mgr, daemon, "foo", None)  # type: ignore[arg-type]
    cleared = await rearm_startup_app_watcher(mgr, daemon, None, first)  # type: ignore[arg-type]

    assert cleared is None
    assert first is not None and first.cancelled()
    # A later wake must not start anything after the startup app is cleared.
    await daemon.backend.wake_up()
    await asyncio.sleep(0)
    assert mgr.started == []


@pytest.mark.asyncio
async def test_rearm_unknown_app_arms_nothing() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    daemon = StubDaemon()

    task = await rearm_startup_app_watcher(mgr, daemon, "nope", None)  # type: ignore[arg-type]
    assert task is None


def test_commanded_motion_detects_moving_target() -> None:
    # Sleep swing: target jumps far between polls -> commanded motion.
    assert _antennas_in_commanded_motion((-0.17, 0.17), (-0.5, 0.5)) is True


def test_commanded_motion_false_for_stable_target() -> None:
    # Idle target barely changes (below eps) -> not commanded motion, so a
    # physical push (present-only) can still be detected.
    assert _antennas_in_commanded_motion((-0.17, 0.17), (-0.17, 0.171)) is False


def test_commanded_motion_false_when_sample_missing() -> None:
    assert _antennas_in_commanded_motion(None, (0.0, 0.0)) is False
    assert _antennas_in_commanded_motion((0.0, 0.0), None) is False


@pytest.mark.asyncio
async def test_remove_app_clears_startup_app(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup_app_config.set_startup_app("foo")
    monkeypatch.setattr(apps_router.bg_job_register, "run_command", lambda *a, **k: "j1")
    request = _stub_request(StubDaemon())
    await apps_router.remove_app(
        "foo", request, StubAppManager(installed=["foo"], catalog=[])  # type: ignore[arg-type]
    )
    assert startup_app_config.get_startup_app() is None
    assert request.app.state.startup_app_antenna_watcher_task is None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_remove_other_app_keeps_startup_app(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup_app_config.set_startup_app("foo")
    monkeypatch.setattr(apps_router.bg_job_register, "run_command", lambda *a, **k: "j1")
    request = _stub_request(StubDaemon())
    await apps_router.remove_app(
        "bar", request, StubAppManager(installed=["foo", "bar"], catalog=[])  # type: ignore[arg-type]
    )
    assert startup_app_config.get_startup_app() == "foo"
