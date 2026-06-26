"""Utility functions for running shell commands asynchronously with real-time logging."""

import asyncio
import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Callable

GITHUB_REPO = "pollen-robotics/reachy_mini"

logger = logging.getLogger(__name__)


def _check_uv_available() -> bool:
    """Check if uv is available on the system."""
    return shutil.which("uv") is not None


def build_install_command(
    extras: str,
    *,
    git_ref: str | None = None,
    version: str | None = None,
    pre_release: bool = False,
    python: Path | None = None,
    upgrade: bool = False,
    verbose: bool = False,
) -> tuple[str, dict[str, str]]:
    """Build a pip/uv install shell command for reachy-mini.

    For a *git_ref* install the command chains three steps:

    1. ``--force-reinstall --no-deps --no-cache-dir`` - reinstall the package
       itself without the dependencies.
    2. ``check`` - check if dependencies need to be updated.
    3. ``--upgrade --upgrade-strategy only-if-needed`` - if dependencies need to be updated, upgrade them.

    Args:
        extras: Pip extras string, e.g. ``"gstreamer"`` or
            ``"wireless-version, gstreamer"``.
        git_ref: If set, install from this GitHub tag/branch.
        version: If set (and *git_ref* is ``None``), pin to this PyPI version.
        pre_release: If ``True`` (and neither *git_ref* nor *version* is set),
            add ``--pre`` to allow pre-release versions.
        python: Target Python interpreter for an external venv.
            Uses ``uv`` when available, otherwise the venv's own ``pip``.
            If ``None``, uses the current environment's ``pip``.
        upgrade: If ``True``, add the ``--upgrade`` flag.
        verbose: If ``True``, add ``-vvv`` flag (only on step 2 for git ref).

    Returns:
        A tuple of the form ``(command, extra_env)`` where *command* is a shell string and *extra_env* contains any additional environment variables needed for the install.

    """
    # --- Base command (pip or uv) ---
    if python is not None:
        if _check_uv_available():
            base = ["uv", "pip", "install", "--python", str(python)]
            logger.info(f"Using uv with python: {python}")
        else:
            base = [str(python.parent / "pip"), "install"]
            logger.info(f"Using pip from venv: {python.parent / 'pip'}")
    else:
        if _check_uv_available():
            base = ["uv", "pip", "install"]
            logger.info("Using uv pip with current environment")
        else:
            base = ["pip", "install"]
            logger.info("Using pip with current environment")

    if verbose:
        base.append("-vvv")

    # --- Package, extra args & env ---
    if git_ref:
        logger.info(f"Installing from git ref: {git_ref}")
        git_url = f"git+https://github.com/{GITHUB_REPO}.git@{git_ref}"
        git_package = f"reachy-mini[{extras}] @ {git_url}"
        # Step 1: force reinstall the package without the dependencies
        step1 = shlex.join(base + [git_package, "--force-reinstall", "--no-deps", "--no-cache-dir"])
        # Step 2: check if dependencies need to be updated
        check_base = [arg if arg != "install" else "check" for arg in base]
        step2 = shlex.join(check_base)
        # Step 3: update dependencies if needed
        step3_args = base + [f"reachy-mini[{extras}]", "--upgrade"]
        if not _check_uv_available():
            step3_args += ["--upgrade-strategy", "only-if-needed"]
        step3 = shlex.join(step3_args)
        cmd = f"{step1} && {step2} || {step3}"
        logger.info(f"Git ref install: {cmd}")
        extra_env: dict[str, str] = {"GIT_LFS_SKIP_SMUDGE": "1"}
        return cmd, extra_env

    logger.info(f"Installing from PyPI: {version if version else 'latest pre-release' if pre_release else 'latest stable'}")
    package = f"reachy-mini[{extras}]"
    if version:
        package = f"{package}=={version}"
    extra_args = []
    if pre_release:
        extra_args.append("--pre")
    if upgrade:
        extra_args.append("--upgrade")
    extra_env = {}

    cmd = shlex.join(base + [package] + extra_args)
    logger.info(f"Install command: {cmd}")
    return cmd, extra_env


async def call_logger_wrapper(
    command: str,
    logger: logging.Logger,
    env: dict[str, str] | None = None,
) -> None:
    """Run a shell command asynchronously, streaming stdout and stderr to logger in real time.

    Args:
        command: Shell command string.
        logger: logger object with .info and .error methods
        env: Optional environment variables dict. If None, inherits current environment.

    """
    logger.info(f"Running: {command}")
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **env} if env else None,
        cwd=Path.home(),
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
