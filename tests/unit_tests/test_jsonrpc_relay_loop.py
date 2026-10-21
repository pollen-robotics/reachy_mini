"""The JSON-RPC relay must survive a daemon started over HTTP.

`POST /api/daemon/start` runs `Daemon.start` as a background job, and those get
a throwaway event loop (`asyncio.run` in a thread) that is closed as soon as the
job finishes. A relay scheduled onto that loop stops answering, so every
`rpcCall` from a phone times out until the process restarts.
"""

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from reachy_mini.daemon.daemon import Daemon


class _Backend:
    """Minimal stand-in for the bits `_setup_jsonrpc_relay` touches."""

    def __init__(self) -> None:
        self.handler: Any = None

    def broadcast_to_all_clients(self, _message: str) -> None:
        """Ignore notifications; this test only cares about replies."""

    def set_jsonrpc_handler(self, handler: Any) -> None:
        """Record the handler the daemon wires into the DataChannel."""
        self.handler = handler


def _app_manager() -> SimpleNamespace:
    """Build an AppManager with nothing running, so `apps.status` answers `idle`."""
    return SimpleNamespace(current_app=None, is_app_running=lambda: False)


def _call(backend: _Backend, method: str) -> dict[str, Any]:
    """Send one frame the way the media thread does, and wait for the reply."""
    answered = threading.Event()
    replies: list[dict[str, Any]] = []

    def reply(response: dict[str, Any]) -> None:
        replies.append(response)
        answered.set()

    frame = json.dumps({"jsonrpc": "2.0", "id": "1", "method": method, "params": {}})
    threading.Thread(target=lambda: backend.handler(frame, reply)).start()
    assert answered.wait(timeout=5), f"{method} was never answered"
    return replies[0]


@pytest.mark.asyncio
async def test_relay_answers_after_a_start_on_a_throwaway_loop() -> None:
    """Wiring from a background job must still answer on the app's own loop."""
    daemon = Daemon(no_media=True)
    daemon.set_rpc_loop(asyncio.get_running_loop())
    backend = _Backend()

    # Exactly what bg_job_register does for `POST /api/daemon/start`.
    def run_start_job() -> None:
        asyncio.run(_wire(daemon, backend))

    job = threading.Thread(target=run_start_job)
    job.start()
    job.join(timeout=5)
    assert not job.is_alive()

    # The job's loop is closed by now; the relay must not have been bound to it.
    response = await asyncio.to_thread(_call, backend, "apps.status")
    assert response["result"]["state"] == "idle"


async def _wire(daemon: Daemon, backend: Any) -> None:
    """Stand in for the part of `Daemon.start` that wires the relay."""
    daemon._setup_jsonrpc_relay(backend, _app_manager())


@pytest.mark.asyncio
async def test_a_closed_loop_is_refused_instead_of_dropped() -> None:
    """On shutdown the caller gets an error, not its full timeout of silence."""
    daemon = Daemon(no_media=True)
    doomed = asyncio.new_event_loop()
    doomed.close()
    daemon.set_rpc_loop(doomed)
    backend = _Backend()
    daemon._setup_jsonrpc_relay(backend, _app_manager())

    response = await asyncio.to_thread(_call, backend, "apps.status")

    assert response["error"]["data"]["reason"] == "relay_unavailable"
    assert response["id"] == "1"
