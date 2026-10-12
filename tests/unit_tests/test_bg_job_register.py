"""Regression tests for background job log streaming over WebSocket."""

import asyncio
import json
import threading

import pytest

from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.app.bg_job_register import JobStatus


class _FakeWebSocket:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, text: str) -> None:
        self.texts.append(text)

    async def send_json(self, data: dict) -> None:  # pragma: no cover
        self.texts.append(json.dumps(data))


@pytest.mark.asyncio
async def test_ws_poll_info_receives_terminal_status_from_worker_thread():
    """A job finishing in its worker thread must wake the WS waiter in the main loop.

    ``run_command`` runs the job in a separate thread with its own event loop, but
    the ``new_log_evt`` waiters live in the daemon loop. ``asyncio.Event.set()`` is
    not thread-safe, so without hopping back onto the daemon loop the terminal DONE
    notification is lost and ``ws_poll_info`` hangs forever (the desktop app's
    "finalizing" step never completes and the connection eventually drops).
    """
    proceed = threading.Event()

    async def job(logger):
        logger.info("working")
        await asyncio.to_thread(proceed.wait)

    job_id = bg_job_register.run_command("test", job)

    ws = _FakeWebSocket()
    task = asyncio.create_task(bg_job_register.ws_poll_info(ws, job_id))

    await asyncio.sleep(0.1)  # let ws_poll_info register its waiter
    proceed.set()

    await asyncio.wait_for(task, timeout=10)

    terminal = json.loads(ws.texts[-1])
    assert terminal["status"] == JobStatus.DONE.value
    assert any("completed successfully" in t for t in ws.texts)
