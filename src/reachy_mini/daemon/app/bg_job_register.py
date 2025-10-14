"""Background jobs management for Reachy Mini Daemon."""

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel


class JobStatus(Enum):
    """Enum for job status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class JobInfo(BaseModel):
    """Pydantic model for install job status."""

    command: str
    status: JobStatus
    logs: list[str]


@dataclass
class JobHandler:
    """Handler for background jobs."""

    uuid: str
    info: JobInfo
    new_log_evt: dict[str, asyncio.Event]


register: dict[str, JobHandler] = {}


def run_command(
    command: str,
    coro_func: Callable[..., Awaitable[None]],
    *args: Any,
) -> str:
    """Start a background job, with a custom logger and return its job_id."""
    job_uuid = str(uuid.uuid4())

    jh = JobHandler(
        uuid=job_uuid,
        info=JobInfo(command=command, status=JobStatus.PENDING, logs=[]),
        new_log_evt={},
    )
    register[job_uuid] = jh

    start_evt = threading.Event()

    async def wrapper() -> None:
        jh.info.status = JobStatus.IN_PROGRESS

        class JobLogger(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                jh.info.logs.append(self.format(record))
                for ws in jh.new_log_evt.values():
                    ws.set()

        logger = logging.getLogger(f"logs_job_{job_uuid}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(JobLogger())

        start_evt.set()

        try:
            await coro_func(*args, logger=logger)
            jh.info.status = JobStatus.DONE
            logger.info(f"Job '{command}' completed successfully")
        except Exception as e:
            jh.info.status = JobStatus.FAILED
            logger.error(f"Job '{command}' failed with error: {e}")

    t = threading.Thread(target=lambda: asyncio.run(wrapper()))
    t.start()
    # background_tasks.add_task(wrapper)
    start_evt.wait()

    return job_uuid


def get_info(job_id: str) -> JobInfo:
    """Get the info of a job by its ID."""
    job = register.get(job_id)

    if not job:
        raise ValueError("Job ID not found")

    return job.info


async def ws_poll_info(websocket: WebSocket, job_uuid: str) -> None:
    """WebSocket endpoint to stream job logs in real time."""
    job = register.get(job_uuid)
    if not job:
        await websocket.send_json({"error": "Job ID not found"})
        await websocket.close()
        return

    assert job is not None

    ws_uuid = str(uuid.uuid4())
    last_log_len = 0

    try:
        job.new_log_evt[ws_uuid] = asyncio.Event()

        while True:
            await job.new_log_evt[ws_uuid].wait()
            job.new_log_evt[ws_uuid].clear()

            new_logs = job.info.logs[last_log_len:]

            if new_logs:
                for log_entry in new_logs:
                    await websocket.send_text(log_entry)
                last_log_len = len(job.info.logs)

                await websocket.send_text(
                    JobInfo(
                        command=job.info.command,
                        status=job.info.status,
                        logs=new_logs,
                    ).model_dump_json()
                )
                if job.info.status in (JobStatus.DONE, JobStatus.FAILED):
                    break
    except WebSocketDisconnect:
        pass
    finally:
        job.new_log_evt.pop(ws_uuid, None)
