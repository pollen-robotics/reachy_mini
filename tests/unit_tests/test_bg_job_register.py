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
@pytest.mark.parametrize("from_threadpool", [False, True])
async def test_ws_poll_info_receives_terminal_status_from_worker_thread(
    from_threadpool: bool,
):
    """A job finishing in its worker thread must wake the WS waiter in the main loop.

    ``run_command`` runs the job in a separate thread with its own event loop, but
    the ``new_log_evt`` waiters live in the daemon loop. ``asyncio.Event.set()`` is
    not thread-safe, so without hopping back onto the daemon loop the terminal DONE
    notification is lost and ``ws_poll_info`` hangs forever (the desktop app's
    "finalizing" step never completes and the connection eventually drops).
    ``from_threadpool`` mimics sync routes (``/update/start``), which call
    ``run_command`` from FastAPI's threadpool, with no running loop.
    """
    proceed = threading.Event()

    async def job(logger):
        logger.info("working")
        await asyncio.to_thread(proceed.wait)

    if from_threadpool:
        job_id = await asyncio.to_thread(bg_job_register.run_command, "test", job)
    else:
        job_id = bg_job_register.run_command("test", job)

    ws = _FakeWebSocket()
    task = asyncio.create_task(bg_job_register.ws_poll_info(ws, job_id))

    await asyncio.sleep(0.1)  # let ws_poll_info register its waiter
    proceed.set()

    await asyncio.wait_for(task, timeout=10)

    terminal = json.loads(ws.texts[-1])
    assert terminal["status"] == JobStatus.DONE.value
    assert any("completed successfully" in t for t in ws.texts)


@pytest.mark.asyncio
async def test_ws_poll_info_joining_after_job_ended():
    """A WS that connects after the job ended still gets the logs and terminal status."""

    async def job(logger):
        raise RuntimeError("boom")

    job_id = bg_job_register.run_command("test", job)
    while bg_job_register.get_info(job_id).status != JobStatus.FAILED:
        await asyncio.sleep(0.01)

    ws = _FakeWebSocket()
    await asyncio.wait_for(bg_job_register.ws_poll_info(ws, job_id), timeout=5)

    terminal = json.loads(ws.texts[-1])
    assert terminal["status"] == JobStatus.FAILED.value
    assert any("boom" in t for t in ws.texts)


@pytest.mark.asyncio
async def test_ws_poll_info_status_flips_while_sending():
    """The job ending mid-send must not stop the stream before DONE is sent.

    Seen on a Wireless: a remove job ended while ``ws_poll_info`` was sending
    an ``in_progress`` update; it re-read the status, broke out, and the
    client never got DONE.
    """
    proceed = threading.Event()

    async def job(logger):
        logger.info("working")
        await asyncio.to_thread(proceed.wait)

    job_id = bg_job_register.run_command("test", job)

    class _JobEndsDuringSend(_FakeWebSocket):
        async def send_text(self, text: str) -> None:
            await super().send_text(text)
            if text.startswith("{") and not proceed.is_set():
                # status flips first, the final log line follows
                bg_job_register.get_info(job_id).status = JobStatus.DONE
                proceed.set()

    ws = _JobEndsDuringSend()
    await asyncio.wait_for(bg_job_register.ws_poll_info(ws, job_id), timeout=5)

    terminal = json.loads(ws.texts[-1])
    assert terminal["status"] == JobStatus.DONE.value
