import asyncio
import threading
from pathlib import Path
from threading import Event
import time
import pytest
import uvicorn

from reachy_mini import ReachyMiniApp
from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.manager import AppManager, AppState
from reachy_mini.apps.sources import local_common_venv
from reachy_mini.daemon.app.main import Args, create_app
from reachy_mini.daemon.daemon import Daemon
from reachy_mini.reachy_mini import ReachyMini


@pytest.mark.asyncio
async def test_app() -> None:
    class MockApp(ReachyMiniApp):
        def run(self, reachy_mini: ReachyMini, stop_event: Event) -> None:
            time.sleep(1)  # Simulate some processing time

    args = Args(
        sim=True,
        headless=True,
        wake_up_on_start=False,
        no_media=True,
        autostart=True,
        fastapi_port=0,
    )
    app = create_app(args)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    while not server.started:
        await asyncio.sleep(0.05)

    sockets = server.servers[0].sockets  # type: ignore[union-attr]
    port: int = sockets[0].getsockname()[1]
    daemon = app.state.daemon

    stop = Event()

    with ReachyMini(host="localhost", port=port, media_backend="no_media") as mini:
        mock_app = MockApp()
        mock_app.run(mini, stop)

    await daemon.stop(goto_sleep_on_stop=False)
    server.should_exit = True
    server_thread.join(timeout=10)


@pytest.mark.asyncio
async def test_app_manager() -> None:
    daemon = Daemon(no_media=True)
    await daemon.start(
        sim=True,
        headless=True,
        wake_up_on_start=False,
        use_audio=False,
    )

    app_mngr = AppManager()
    try:
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
        assert status is not None and status.state in (
            AppState.STARTING,
            AppState.RUNNING,
        )
        assert app_mngr.is_app_running()
        status = await app_mngr.current_app_status()
        assert status is not None and status.state in (
            AppState.STARTING,
            AppState.RUNNING,
        )

        await app_mngr.stop_current_app()
        assert not app_mngr.is_app_running()
        status = await app_mngr.current_app_status()
        assert status is None

        await app_mngr.remove_app("ok_app", daemon.logger)
        after_uninstalled_apps = await app_mngr.list_available_apps(
            SourceKind.INSTALLED
        )

        assert len(after_uninstalled_apps) == len(before_installed_apps)

    except Exception as e:
        pytest.fail(f"install_new_app raised an exception: {e}")
    finally:
        await daemon.stop(goto_sleep_on_stop=False)


@pytest.mark.asyncio
async def test_faulty_app() -> None:
    # Start a real app server on port 8000 so the faulty app subprocess
    # can connect immediately (its _check_daemon_on_localhost checks port 8000).
    # Without a server, the subprocess falls back to reachy-mini.local DNS
    # resolution which hangs in CI, causing the test to time out.
    args = Args(
        sim=True,
        headless=True,
        wake_up_on_start=False,
        no_media=True,
        autostart=True,
        fastapi_port=8000,
    )
    app = create_app(args)
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    while not server.started:
        await asyncio.sleep(0.05)

    daemon = app.state.daemon
    app_mngr = AppManager()

    app_info = AppInfo(
        name="faulty_app",
        source_kind=SourceKind.LOCAL,
        extra={"path": str(Path(__file__).parent / "faulty_app")},
    )
    try:
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

        if not success:
            pytest.fail("Faulty app did not reach ERROR state in time")

    except Exception as e:
        pytest.fail(f"install_new_app raised an exception: {e}")
    finally:
        await daemon.stop(goto_sleep_on_stop=False)
        server.should_exit = True
        server_thread.join(timeout=10)


@pytest.mark.asyncio
async def test_start_app_scrubs_leaking_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app subprocess must not inherit env vars pointing at the daemon's venv.

    PYTHONPATH and GST_PYTHONPATH_1_0 are set by the daemon's own
    gstreamer_bundle.pth and take precedence over the target interpreter's
    site-packages. Leaking them makes an app launched with apps_venv's Python
    import `reachy_mini` and `gi` from the daemon's venv instead, which breaks
    whenever the two environments hold different SDK versions.
    """
    leaking = {
        "PYTHONPATH": "/daemon_venv/lib/python3.12/site-packages",
        "GST_PYTHONPATH_1_0": "/daemon_venv/lib/python3.12/site-packages/gstreamer_python",
        "GST_REGISTRY_1_0": "/daemon_venv/registry.bin",
        "XDG_DATA_DIRS": "/daemon_venv/share",
    }
    for key, value in leaking.items():
        monkeypatch.setenv(key, value)

    captured_env: dict[str, str] = {}

    class _FakeProcess:
        returncode = 0

        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        async def wait(self) -> int:
            return 0

    async def fake_exec(*args: object, **kwargs: object) -> _FakeProcess:
        captured_env.update(kwargs["env"])  # type: ignore[arg-type]
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        local_common_venv, "get_app_module", lambda *a, **k: "some_app.main"
    )
    monkeypatch.setattr(
        local_common_venv, "get_app_python", lambda *a, **k: "/apps_venv/bin/python"
    )

    app_mngr = AppManager()
    await app_mngr.start_app("some_app")
    try:
        assert captured_env, "subprocess was never launched"
        for key in leaking:
            assert key not in captured_env, (
                f"{key} leaked into the app subprocess environment"
            )
        # A neutral variable must still be forwarded.
        assert "PATH" in captured_env
    finally:
        await app_mngr.stop_current_app()
