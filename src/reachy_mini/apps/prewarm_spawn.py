"""Early parked-app spawn: overlap the app's import phase with the daemon's.

On the wireless unit the daemon spends ~9s importing before its lifespan can
spawn the parked app, whose own ~5s import phase then lands on the cold-boot
critical path (the boot auto-start waits on it). `spawn_early_parked_app` is
called at daemon-entry time, BEFORE the daemon's heavy imports, so both import
phases run in parallel on separate cores. The lifespan later hands the process
to `AppManager.adopt_early_parked_app`, which wraps it in `AdoptedPopen` so
the existing parked machinery (drain/activate/evict) sees the asyncio
subprocess interface it expects.

Everything here must stay stdlib-light (no fastapi, no huggingface_hub, no
reachy_mini.apps.app) and fail-safe: any doubt returns None and the normal
keeper path takes over.
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

from reachy_mini.apps.sources import local_venv_paths
from reachy_mini.daemon import startup_app_config

logger = logging.getLogger(__name__)

# Duplicated from reachy_mini.apps.app (which imports the full SDK — exactly
# what this early path must avoid). Pinned by tests in test_prewarm_spawn.py.
PARKED_ENV_VAR = "REACHY_MINI_START_PARKED"
PARKED_READY_SENTINEL = "REACHY_MINI_APP_PARKED_READY"

# Mirrors startup_check.STAMP_PATH (stdlib module, but under utils/ whose
# __init__ pulls numpy; keep the literal instead). Pinned by a test.
STAMP_PATH = Path("/venvs/mini_daemon") / ".startup_check_stamp.json"

# GStreamer env vars the daemon's venv sets that must not leak into an app
# subprocess running in apps_venv (see manager._build_app_env history and
# pollen-robotics/reachy-mini-desktop-app#185).
_SCRUBBED_ENV_KEYS = (
    "GST_PLUGIN_PATH_1_0",
    "GST_PLUGIN_SYSTEM_PATH_1_0",
    "GST_REGISTRY_1_0",
    "GST_PLUGIN_SCANNER_1_0",
    "GI_TYPELIB_PATH",
    "PYGI_DLL_DIRS",
    "XDG_DATA_DIRS",
    "XDG_CONFIG_DIRS",
)


def build_app_env(*, parked: bool) -> "dict[str, str]":
    """Environment for an app subprocess (shared with AppManager).

    Scrub GStreamer env vars that the daemon's own ``gstreamer_bundle.pth``
    set pointing at paths inside the daemon's venv; the app venv sets fresh
    values at Python startup and the parent's values are actively harmful.
    """
    app_env = os.environ.copy()
    for key in _SCRUBBED_ENV_KEYS:
        app_env.pop(key, None)
    if parked:
        app_env[PARKED_ENV_VAR] = "1"
    return app_env


def spawn_early_parked_app(
    argv: "list[str]",
) -> "tuple[subprocess.Popen[bytes], str] | None":
    """Spawn the parked prewarm app before the daemon's heavy imports.

    Gates (all fail-safe to None = today's behavior):
    - wireless daemon with autostart (``--wireless-version``, no
      ``--no-autostart``) — the only mode whose lifespan adopts the process;
    - startup-check stamp present — no stamp means the full checks may
      repair/mutate the apps venv this boot, don't race them;
    - a prewarm app is configured and opts into parking.

    The returned process is parked on stdin; if the daemon dies before
    adoption, the dropped pipe EOFs it into a cooperative exit(0).
    """
    try:
        if "--wireless-version" not in argv or "--no-autostart" in argv:
            return None
        if not STAMP_PATH.exists():
            return None
        name = startup_app_config.get_prewarm_app()
        if not name:
            return None
        if not local_venv_paths.app_supports_parking(name, wireless_version=True):
            return None
        module = local_venv_paths.get_app_module(name, wireless_version=True)
        python = local_venv_paths.get_app_python(name, wireless_version=True)
        process = subprocess.Popen(
            [str(python), "-u", "-m", module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=build_app_env(parked=True),
        )
        logger.info(f"Early parked-app spawn: '{name}' (pid {process.pid})")
        # Logging is not configured yet at daemon entry; stderr still reaches
        # the journal, so measurements get a timestamped spawn marker.
        print(
            f"[early-prewarm] spawned '{name}' (pid {process.pid})",
            file=sys.stderr,
            flush=True,
        )
        return process, name
    except Exception:
        logger.warning("Early parked-app spawn failed; keeper will spawn later",
                       exc_info=True)
        return None


class AdoptedPopen:
    """asyncio-Process-shaped adapter around an early-spawned Popen.

    Exposes the surface the parked/runner machinery uses on an
    ``asyncio.subprocess.Process``: ``stdin`` (StreamWriter), ``stdout`` /
    ``stderr`` (StreamReader), ``pid``, ``returncode``, ``wait()``,
    ``kill()``, ``terminate()``, ``send_signal()``. ``connect_streams`` must
    be awaited once (inside a running loop) before the streams are used.
    """

    def __init__(self, popen: "subprocess.Popen[bytes]") -> None:
        """Wrap ``popen``; call ``connect_streams`` before using the streams."""
        self._popen = popen
        self.pid = popen.pid
        self.stdin: asyncio.StreamWriter | None = None
        self.stdout: asyncio.StreamReader | None = None
        self.stderr: asyncio.StreamReader | None = None

    async def connect_streams(self) -> None:
        """Wrap the Popen pipes into asyncio streams on the running loop."""
        loop = asyncio.get_running_loop()

        async def reader(pipe: object) -> asyncio.StreamReader:
            stream = asyncio.StreamReader()
            await loop.connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(stream), pipe
            )
            return stream

        assert self._popen.stdout is not None
        assert self._popen.stderr is not None
        assert self._popen.stdin is not None
        self.stdout = await reader(self._popen.stdout)
        self.stderr = await reader(self._popen.stderr)
        transport, protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, self._popen.stdin
        )
        self.stdin = asyncio.StreamWriter(transport, protocol, None, loop)

    @property
    def returncode(self) -> "int | None":
        """Exit code if the process has finished, else None (poll semantics)."""
        return self._popen.poll()

    async def wait(self) -> int:
        """Await process exit off-loop (Popen.wait is thread-safe)."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._popen.wait
        )

    def kill(self) -> None:
        """Forward to Popen.kill."""
        self._popen.kill()

    def terminate(self) -> None:
        """Forward to Popen.terminate."""
        self._popen.terminate()

    def send_signal(self, sig: int) -> None:
        """Forward to Popen.send_signal."""
        self._popen.send_signal(sig)
