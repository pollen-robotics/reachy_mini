
import asyncio
from pathlib import Path
from threading import Event
import time
import pytest

from reachy_mini import ReachyMiniApp
from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.manager import AppManager, AppState
from reachy_mini.daemon.daemon import Daemon    
from reachy_mini.reachy_mini import ReachyMini


@pytest.mark.asyncio
async def test_app() -> None: 
    class MockApp(ReachyMiniApp):
        def run(self, reachy_mini: ReachyMini, stop_event: Event) -> None:
            time.sleep(1)  # Simulate some processing time

    daemon = Daemon()
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
    )

    stop = Event()

    with ReachyMini(media_backend="no_media") as mini:
        app = MockApp()
        app.run(mini, stop)

    await daemon.stop(goto_sleep_on_stop=False)


@pytest.mark.asyncio
async def test_app_manager() -> None: 
    daemon = Daemon()
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
    )

    app_mngr = AppManager()

    before_installed_apps = await app_mngr.list_available_apps(SourceKind.INSTALLED)
    
    app_info = AppInfo(
        name="ok_app",
        source_kind=SourceKind.LOCAL,
        extra={"path": str(Path(__file__).parent / "ok_app")},
    )
    await app_mngr.install_new_app(app_info, daemon.logger)

    after_installed_apps = await app_mngr.list_available_apps(SourceKind.INSTALLED)

    assert len(after_installed_apps) == len(before_installed_apps) + 1

    status = await app_mngr.start_app("ok_app", media_backend="no_media")
    assert status is not None and status.state in (AppState.STARTING, AppState.RUNNING)
    assert app_mngr.is_app_running()
    status = await app_mngr.current_app_status()
    assert status is not None and status.state in (AppState.STARTING, AppState.RUNNING)

    await app_mngr.stop_current_app()
    assert not app_mngr.is_app_running()
    status = await app_mngr.current_app_status()
    assert status is None

    await app_mngr.remove_app("ok_app", daemon.logger)
    after_uninstalled_apps = await app_mngr.list_available_apps(SourceKind.INSTALLED)

    assert len(after_uninstalled_apps) == len(before_installed_apps)

    await daemon.stop(goto_sleep_on_stop=False)



@pytest.mark.asyncio
async def test_faulty_app() -> None: 
    daemon = Daemon()
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
    )

    app_mngr = AppManager()
    
    app_info = AppInfo(
        name="faulty_app",
        source_kind=SourceKind.LOCAL,
        extra={"path": str(Path(__file__).parent / "faulty_app")},
    )
    await app_mngr.install_new_app(app_info, daemon.logger)

    status = await app_mngr.start_app("faulty_app", media_backend="no_media")

    success = False
    for _ in range(10):
        status = await app_mngr.current_app_status()
        if status is None or status.state in (AppState.STARTING, AppState.RUNNING):
            await asyncio.sleep(1.0)
            continue

        if status is not None and status.state == AppState.ERROR:
            success = True
            break

    await app_mngr.remove_app("faulty_app", daemon.logger)
    await daemon.stop(goto_sleep_on_stop=False)

    if not success:
        pytest.fail("Faulty app did not reach ERROR state in time")


