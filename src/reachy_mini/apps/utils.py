"""Utility functions for Reachy Mini apps manager."""

import asyncio
import logging
import os
from pathlib import Path


async def running_command(
    command: list[str],
    logger: logging.Logger,
    env: dict[str, str] | None = None,
) -> int:
    """Run a shell command and stream its output to the provided logger.

    Args:
        command: The command to run as a list of strings.
        logger: Logger instance for output streaming.
        env: Optional environment variables dict. If None, inherits current environment.

    """
    logger.info(f"Running command: {' '.join(command)}")

    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **env} if env else None,
        cwd=Path.home(),
    )

    assert proc.stdout is not None  # for mypy
    assert proc.stderr is not None  # for mypy

    # Stream output line by line
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        logger.info(line.decode().rstrip())

    # Also log any remaining stderr
    err = await proc.stderr.read()
    if err:
        logger.error(err.decode().rstrip())

    return await proc.wait()
