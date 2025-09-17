"""Utility functions for Reachy Mini apps manager."""

import asyncio
import logging


async def running_command(command: list[str], logger: logging.Logger) -> int:
    """Run a shell command and stream its output to the provided logger."""
    logger.info(f"Running command: {' '.join(command)}")

    proc = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
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
