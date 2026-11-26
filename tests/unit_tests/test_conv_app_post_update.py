"""The conversation app is updated on the first boot of a new daemon version."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.daemon import startup_app_config
from reachy_mini.daemon.app import startup_app
from reachy_mini.daemon.app.startup_app import (
    CONVERSATION_APP_NAME,
    update_conversation_app_after_daemon_update,
)


class Manager:
    """Serves a canned installed list and records updates."""

    wireless_version = True
    desktop_app_daemon = False

    def __init__(self, installed: list[str], fail_update: bool = False) -> None:
        """Pretend `installed` are the installed apps."""
        self._installed = installed
        self._fail_update = fail_update
        self.updated: list[str] = []
        self.reinstalled: list[str] = []

    async def list_available_apps(self, source: SourceKind) -> list[AppInfo]:
        """Return the installed apps, or the catalog (the conversation app)."""
        names = (
            self._installed
            if source == SourceKind.INSTALLED
            else [CONVERSATION_APP_NAME]
        )
        return [AppInfo(name=n, source_kind=source) for n in names]

    async def update_app(self, name: str, logger: object) -> None:
        """Record the update, or fail it after the uninstall like update_app can."""
        if self._fail_update:
            self._installed.remove(name)
            raise RuntimeError("network down")
        self.updated.append(name)

    async def install_new_app(self, app: AppInfo, logger: object) -> None:
        """Record a reinstall."""
        self.reinstalled.append(app.name)
        self._installed.append(app.name)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Throwaway config file + a controllable Hub answer."""
    path = tmp_path / "daemon_config.json"
    monkeypatch.setattr(startup_app_config, "_config_path", lambda: path)
    state: dict[str, Any] = {
        "from_hub": True,
        "info": SimpleNamespace(update_available=True),
    }

    async def check(app: Any, **kwargs: Any) -> Any:
        return state["info"]

    monkeypatch.setattr(startup_app, "check_app_update", check)
    monkeypatch.setattr(
        startup_app,
        "get_hf_install_info",
        lambda *a: object() if state["from_hub"] else None,
    )
    return state


@pytest.mark.asyncio
async def test_updates_once_per_new_version(hub: dict[str, Any]) -> None:
    """A second boot on the same version does nothing."""
    mgr = Manager([CONVERSATION_APP_NAME])
    await update_conversation_app_after_daemon_update(mgr, "1.16.0")  # type: ignore[arg-type]
    await update_conversation_app_after_daemon_update(mgr, "1.16.0")  # type: ignore[arg-type]
    assert mgr.updated == [CONVERSATION_APP_NAME]
    assert startup_app_config.get_last_booted_version() == "1.16.0"


@pytest.mark.asyncio
async def test_skips_when_up_to_date_or_not_installed(hub: dict[str, Any]) -> None:
    """No update without a newer Hub version or an installed app."""
    hub["info"] = SimpleNamespace(update_available=False)
    mgr = Manager([CONVERSATION_APP_NAME])
    await update_conversation_app_after_daemon_update(mgr, "1.16.0")  # type: ignore[arg-type]
    other = Manager(["some_other_app"])
    await update_conversation_app_after_daemon_update(other, "1.17.0")  # type: ignore[arg-type]
    assert mgr.updated == other.updated == []
    assert startup_app_config.get_last_booted_version() == "1.17.0"


@pytest.mark.asyncio
async def test_retries_next_boot_when_the_hub_is_unreachable(
    hub: dict[str, Any],
) -> None:
    """An unreachable Hub leaves the version unrecorded."""
    hub["info"] = None
    mgr = Manager([CONVERSATION_APP_NAME])
    await update_conversation_app_after_daemon_update(mgr, "1.16.0")  # type: ignore[arg-type]
    assert startup_app_config.get_last_booted_version() is None


@pytest.mark.asyncio
async def test_slow_hub_is_cut_short(
    hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hanging Hub check gives up after the cap and retries next boot."""

    async def hang(app: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(60)

    monkeypatch.setattr(startup_app, "check_app_update", hang)
    monkeypatch.setattr(startup_app, "POST_UPDATE_CHECK_TIMEOUT_S", 0.01)
    mgr = Manager([CONVERSATION_APP_NAME])
    await update_conversation_app_after_daemon_update(mgr, "1.16.0")  # type: ignore[arg-type]
    assert startup_app_config.get_last_booted_version() is None


@pytest.mark.asyncio
async def test_failed_update_reinstalls_the_app(hub: dict[str, Any]) -> None:
    """An update that dies after the uninstall puts the app back."""
    mgr = Manager([CONVERSATION_APP_NAME], fail_update=True)
    await update_conversation_app_after_daemon_update(mgr, "1.16.0")  # type: ignore[arg-type]
    assert mgr.reinstalled == [CONVERSATION_APP_NAME]
    assert startup_app_config.get_last_booted_version() is None


@pytest.mark.asyncio
async def test_leaves_a_local_install_alone(hub: dict[str, Any]) -> None:
    """A local or git install is not ours to update."""
    hub["from_hub"] = False
    mgr = Manager([CONVERSATION_APP_NAME])
    await update_conversation_app_after_daemon_update(mgr, "1.16.0")  # type: ignore[arg-type]
    assert mgr.updated == []
    assert startup_app_config.get_last_booted_version() == "1.16.0"
