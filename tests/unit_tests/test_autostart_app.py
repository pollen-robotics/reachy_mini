import asyncio
from contextlib import suppress
from threading import Event

import pytest

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.daemon.app.startup_app import (
    AntennaTouchDetector,
    ensure_startup_app_installed,
    make_startup_app_launcher,
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
        self.started: list[str] = []
        self.evict_remote_values: list[bool] = []
        self.running = False

    async def list_available_apps(self, source: SourceKind) -> list[AppInfo]:
        names = self._installed if source == SourceKind.INSTALLED else self._catalog
        return [AppInfo(name=n, source_kind=source) for n in names]

    async def install_new_app(self, app: AppInfo, logger: object) -> None:
        self.installed_calls.append(app.name)
        self._installed.append(app.name)

    def is_app_running(self) -> bool:
        return self.running

    async def start_app(self, name: str, *, evict_remote: bool = True) -> None:
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

    def get_present_antenna_joint_positions(self) -> list[float]:
        return self.present

    def get_motor_control_mode(self) -> MotorControlMode:
        return self.motor_control_mode

    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        self.motor_control_mode = mode

    async def wake_up(self) -> None:
        self.wake_up_calls += 1


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

    assert detector.update((0.0, 0.0), (0.0, 0.0)) is False
    assert detector.update((0.20, 0.0), (0.0, 0.0)) is False
    assert detector.update((0.26, 0.0), (0.0, 0.0)) is True
    assert detector.update((0.30, 0.0), (0.0, 0.0)) is False
    assert detector.update((0.05, 0.0), (0.0, 0.0)) is False
    assert detector.update((0.0, -0.26), (0.0, 0.0)) is True


def test_antenna_touch_detector_uses_present_baseline_without_target() -> None:
    detector = AntennaTouchDetector(press_delta_rad=0.25, release_delta_rad=0.1)

    assert detector.update((1.0, -1.0), None) is False
    assert detector.update((1.2, -1.0), None) is False
    assert detector.update((1.3, -1.0), None) is True
    assert detector.update((1.3, -1.0), None) is False
    assert detector.update((1.05, -1.0), None) is False
    assert detector.update((1.0, -1.3), None) is True


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
    assert mgr.started == ["foo"]
    assert mgr.evict_remote_values == [False]


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
