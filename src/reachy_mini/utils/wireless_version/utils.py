"""Utility functions for running shell commands asynchronously with real-time logging."""

import asyncio
import logging
import os
from typing import Callable


async def call_logger_wrapper(
    command: list[str],
    logger: logging.Logger,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command asynchronously, streaming stdout and stderr to logger in real time.

    Args:
        command: list or tuple of command arguments (not a string)
        logger: logger object with .info and .error methods
        env: Optional environment variables dict. If None, inherits current environment.

    """
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **env} if env else None,
    )

    async def stream_output(
        stream: asyncio.StreamReader,
        log_func: Callable[[str], None],
    ) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            log_func(line.decode().rstrip())

    tasks = []
    if process.stdout is not None:
        tasks.append(asyncio.create_task(stream_output(process.stdout, logger.info)))
    if process.stderr is not None:
        tasks.append(asyncio.create_task(stream_output(process.stderr, logger.error)))

    await asyncio.gather(*tasks)
    await process.wait()
