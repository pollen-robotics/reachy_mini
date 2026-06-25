import asyncio

import pytest

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.daemon.app.main import (
    _ensure_startup_app_installed,
    _make_startup_app_launcher,
    _start_startup_app,
)


class StubAppManager:
    """Records install/start calls and serves canned app lists per source."""

    def __init__(self, installed: list[str], catalog: list[str]) -> None:
        self._installed = installed
        self._catalog = catalog
        self.installed_calls: list[str] = []
        self.started: list[str] = []

    async def list_available_apps(self, source: SourceKind) -> list[AppInfo]:
        names = self._installed if source == SourceKind.INSTALLED else self._catalog
        return [AppInfo(name=n, source_kind=source) for n in names]

    async def install_new_app(self, app: AppInfo, logger: object) -> None:
        self.installed_calls.append(app.name)
        self._installed.append(app.name)

    async def start_app(self, name: str) -> None:
        self.started.append(name)


@pytest.mark.asyncio
async def test_already_installed_is_ready_without_install() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=["foo", "bar"])
    assert await _ensure_startup_app_installed(mgr, "foo") is True  # type: ignore[arg-type]
    assert mgr.installed_calls == []


@pytest.mark.asyncio
async def test_missing_app_installed_from_catalog() -> None:
    mgr = StubAppManager(installed=[], catalog=["bar"])
    assert await _ensure_startup_app_installed(mgr, "bar") is True  # type: ignore[arg-type]
    assert mgr.installed_calls == ["bar"]


@pytest.mark.asyncio
async def test_unknown_app_not_ready_and_not_installed() -> None:
    mgr = StubAppManager(installed=[], catalog=["bar"])
    assert await _ensure_startup_app_installed(mgr, "nope") is False  # type: ignore[arg-type]
    assert mgr.installed_calls == []


@pytest.mark.asyncio
async def test_install_failure_returns_false() -> None:
    mgr = StubAppManager(installed=[], catalog=["bar"])

    async def boom(app: AppInfo, logger: object) -> None:
        raise RuntimeError("network down")

    mgr.install_new_app = boom  # type: ignore[assignment]
    assert await _ensure_startup_app_installed(mgr, "bar") is False  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_start_calls_start_app() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    await _start_startup_app(mgr, "foo")  # type: ignore[arg-type]
    assert mgr.started == ["foo"]


@pytest.mark.asyncio
async def test_start_failure_is_swallowed() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])

    async def boom(name: str) -> None:
        raise RuntimeError("an app is already running")

    mgr.start_app = boom  # type: ignore[assignment]
    await _start_startup_app(mgr, "foo")  # must not raise  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_launcher_starts_app_once_across_multiple_wakes() -> None:
    mgr = StubAppManager(installed=["foo"], catalog=[])
    launch = _make_startup_app_launcher(mgr, "foo")  # type: ignore[arg-type]

    # Simulate several wake-ups; only the first should launch the app.
    launch()
    launch()
    launch()
    await asyncio.sleep(0)  # let the scheduled task(s) run

    assert mgr.started == ["foo"]
