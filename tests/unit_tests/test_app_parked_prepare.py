"""Tests for parked-phase preparation of the settings server in wrapped_run.

A parked instance should pay the uvicorn import and server construction
BEFORE blocking on stdin (pure CPU, binds nothing), so activation only has to
start the thread. Eviction (stdin EOF) must still exit without ever binding.
"""

import io
import sys
import types

import pytest

from reachy_mini.apps import app as app_module
from reachy_mini.apps.app import PARKED_ENV_VAR, ReachyMiniApp


class _StubServer:
    """Records lifecycle events instead of binding anything."""

    events: "list[str]" = []

    def __init__(self, config) -> None:
        _StubServer.events.append("server_built")
        self.should_exit = False

    def run(self) -> None:
        _StubServer.events.append("server_run")


@pytest.fixture
def stub_uvicorn(monkeypatch):
    _StubServer.events = []
    stub = types.SimpleNamespace(
        Config=lambda app, host, port: types.SimpleNamespace(),
        Server=_StubServer,
    )
    monkeypatch.setitem(sys.modules, "uvicorn", stub)
    return _StubServer.events


class _SettingsApp(ReachyMiniApp):
    custom_app_url = "http://0.0.0.0:7860/"

    def run(self, reachy_mini, stop_event) -> None:
        pass


def _make_app(monkeypatch) -> _SettingsApp:
    monkeypatch.setattr(
        ReachyMiniApp, "_check_daemon_on_localhost", staticmethod(lambda **k: False)
    )
    app = _SettingsApp()
    app.settings_app = object()  # force the settings-server branch
    return app


def test_parked_eviction_exits_with_server_built_but_never_run(
    monkeypatch, stub_uvicorn
) -> None:
    """The server must be constructed BEFORE parking, and EOF must not run it."""
    monkeypatch.setenv(PARKED_ENV_VAR, "1")
    app = _make_app(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # EOF => evicted

    with pytest.raises(SystemExit):
        app.wrapped_run()

    assert stub_uvicorn == ["server_built"]


def test_activation_starts_server_and_runs_app(monkeypatch, stub_uvicorn) -> None:
    """After the stdin activation line, the server thread starts and run() runs."""
    monkeypatch.setenv(PARKED_ENV_VAR, "1")
    app = _make_app(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO("activate\n"))

    ran = []

    class StubMini:
        def __init__(self, *a, **k) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a) -> None:
            pass

    monkeypatch.setattr(app_module, "ReachyMini", StubMini)
    app.run = lambda mini, stop: ran.append(True)  # type: ignore[method-assign]

    app.wrapped_run()

    assert ran == [True]
    assert "server_built" in stub_uvicorn
    # run() returning sets the stop path; the server thread was started.
    assert "server_run" in stub_uvicorn
