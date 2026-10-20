"""Anonymous daemon telemetry, sent to PostHog EU.

On by default. The choice is ``telemetry_enabled`` in the user-editable
``daemon_config.json`` (also ``PUT /api/telemetry``), and environment
variables can force it off. Events carry a random per-robot id, the form
factor, versions and OS, never IPs, names, Hugging Face identity, hardware
ids, logs or media.
"""

import json
import logging
import os
import platform
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any

import platformdirs
import psutil
import requests

from reachy_mini.daemon import startup_app_config

logger = logging.getLogger(__name__)

POSTHOG_URL = "https://eu.i.posthog.com/batch/"
# Public, write-only project key, shared with the desktop app.
POSTHOG_KEY = "phc_oFlHvjvOT6aWXQ4Fot7A1VSAOHtGv9L2M9BZRZcyYQm"
DEFAULT_ENABLED = True
# Written by the ReachyMiniOS image build ("ReachyMiniOS: <tag>").
OS_IMAGE_VERSION_PATH = Path("/home/pollen/VERSION.txt")

FLUSH_INTERVAL_S = 60.0
HEARTBEAT_INTERVAL_S = 24 * 3600.0
_MAX_QUEUE = 500
_MAX_CLIENTS_SEEN = 50
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})

CLIENT_HEADER = "X-Reachy-Client"
CLIENT_TYPES = frozenset({"desktop", "mobile", "web", "python_sdk"})
CLIENT_OSES = frozenset({"macos", "windows", "linux", "ios", "android"})
_CLIENT_RE = re.compile(
    r"^([a-z_]{1,16})/([0-9A-Za-z.+-]{1,32})(?: \(([a-z]{1,16})\))?$"
)


def _state_path() -> Path:
    # Internal state (robot id...), not user config: the choice is in daemon_config.json.
    return Path(platformdirs.user_config_dir("reachy_mini")) / "telemetry_state.json"


def _read_state() -> dict[str, Any]:
    try:
        data = json.loads(_state_path().read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable telemetry state: %s", e)
        return {}


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def _forget() -> None:
    """Delete the robot id, so a later opt-in starts an unlinkable identity."""
    try:
        _state_path().unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Could not delete telemetry state: %s", e)


def disabled_by_env() -> bool:
    """Return True if an environment variable forbids telemetry."""
    env = {
        k: os.environ.get(k, "").strip().lower()
        for k in ("REACHY_MINI_TELEMETRY", "DO_NOT_TRACK", "HF_HUB_DISABLE_TELEMETRY")
    }
    return (
        env["REACHY_MINI_TELEMETRY"] in _FALSY
        or env["DO_NOT_TRACK"] in _TRUTHY
        or env["HF_HUB_DISABLE_TELEMETRY"] in _TRUTHY
    )


def user_enabled() -> bool:
    """Return the choice from daemon_config.json, or the default when unset."""
    value = startup_app_config.get_telemetry_enabled()
    return DEFAULT_ENABLED if value is None else value


def _os_image_version() -> str | None:
    try:
        for line in OS_IMAGE_VERSION_PATH.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "ReachyMiniOS" and value.strip():
                return value.strip()
    except OSError:
        pass
    return None


def os_fields(form_factor: str) -> dict[str, str]:
    """Return ``os_type`` and ``os_version``: the OS image on Wireless, the host OS otherwise."""
    image_version = _os_image_version() if form_factor == "wireless" else None
    if image_version:
        return {"os_type": "reachy_mini_os", "os_version": image_version}

    system = platform.system().lower()
    if system == "darwin":
        return {"os_type": "macos", "os_version": platform.mac_ver()[0]}
    if system == "windows":
        # release() alone can't tell 10 from 11 on older Pythons; the build can.
        version_str = f"{platform.release()} ({platform.version()})"
        return {"os_type": "windows", "os_version": version_str}
    if system == "linux":
        try:
            info = platform.freedesktop_os_release()
            release = info.get("VERSION_ID") or info.get("VERSION_CODENAME", "")
            version_str = f"{info.get('ID', 'linux')} {release}".strip()
        except OSError:
            version_str = platform.release()
        return {"os_type": "linux", "os_version": version_str}
    return {"os_type": system, "os_version": platform.release()}


def parse_client_header(raw: str) -> dict[str, str] | None:
    """Parse ``<type>/<version> (<os>)``; unknown values become ``other``."""
    match = _CLIENT_RE.match(raw.strip())
    if match is None:
        return None
    client_type, client_version, client_os = match.groups()
    return {
        "client_type": client_type if client_type in CLIENT_TYPES else "other",
        "client_version": client_version,
        "client_os": client_os if client_os in CLIENT_OSES else "other",
    }


def app_id_for(metadata: dict[str, Any]) -> str:
    """Return the public Space id, or ``private`` unless provably public."""
    space_id = metadata.get("id")
    if metadata.get("private") is False and isinstance(space_id, str):
        return space_id
    return "private"


def _install_source() -> str:
    # Not update_available.get_install_source: that module needs semver,
    # which only the wireless extra installs.
    try:
        direct_url = json.loads(
            distribution("reachy_mini").read_text("direct_url.json") or "{}"
        )
    except (OSError, ValueError):
        return "pypi"
    if direct_url.get("dir_info", {}).get("editable"):
        return "editable"
    return "git" if "vcs_info" in direct_url else "pypi"


def _envelope(form_factor: str, desktop_app_daemon: bool) -> dict[str, Any]:
    return {
        "event_source": "daemon",
        "form_factor": form_factor,
        "daemon_version": version("reachy_mini"),
        "install_source": _install_source(),
        **os_fields(form_factor),
        "arch": platform.machine().lower(),
        "desktop_app_daemon": desktop_app_daemon,
        "$geoip_disable": True,
        "$process_person_profile": False,
    }


class Telemetry:
    """Queue events in memory and flush them from a background thread."""

    def __init__(self) -> None:
        """Create an inactive instance; ``start`` activates it."""
        self._lock = threading.Lock()
        self._queue: deque[dict[str, Any]] = deque(maxlen=_MAX_QUEUE)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._context: dict[str, Any] | None = None
        self._envelope: dict[str, Any] | None = None
        self._robot_id: str | None = None
        self._clients_seen: set[tuple[str, ...]] = set()

    @property
    def active(self) -> bool:
        """Return True while events are being collected."""
        return self._envelope is not None

    def start(self, *, form_factor: str, desktop_app_daemon: bool) -> None:
        """Activate at daemon startup if the user and environment allow it."""
        self._context = {
            "form_factor": form_factor,
            "desktop_app_daemon": desktop_app_daemon,
        }
        if disabled_by_env():
            return
        if startup_app_config.get_telemetry_enabled() is None:
            # Write the default so users find the key in their config file.
            try:
                startup_app_config.set_telemetry_enabled(DEFAULT_ENABLED)
            except OSError as e:
                logger.warning("Could not save the telemetry default: %s", e)
        if not user_enabled():
            _forget()  # the file may have been hand-edited to false
            return
        ready_ms = int((time.time() - psutil.Process().create_time()) * 1000)
        self._activate(time_to_ready_ms=ready_ms)

    def stop(self) -> None:
        """Flush what is pending and mark a clean daemon exit."""
        if not self.active:
            return
        self._halt(flush=True)
        try:
            _write_state({**_read_state(), "running": False})
        except OSError as e:
            logger.warning("Could not save telemetry state: %s", e)

    def set_enabled(self, enabled: bool) -> None:
        """Persist the user's choice and apply it live.

        Disabling also forgets the robot id, so re-enabling starts a new,
        unlinkable identity. May raise OSError on a write error.
        """
        startup_app_config.set_telemetry_enabled(enabled)
        if enabled and not self.active and self._context and not disabled_by_env():
            self._activate(time_to_ready_ms=None)
        elif not enabled:
            if self.active:
                self._halt(flush=False)
            _forget()

    def capture(self, event: str, **properties: Any) -> None:
        """Queue an event; a no-op while inactive. Never blocks on I/O."""
        with self._lock:
            if self._envelope is None:
                return
            self._queue.append(
                {
                    "event": event,
                    "distinct_id": self._robot_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "properties": {**self._envelope, **properties},
                }
            )

    def note_client(self, raw_header: str) -> None:
        """Emit ``client_connected`` once per distinct client per daemon run."""
        client = parse_client_header(raw_header)
        if client is None or not self.active:
            return
        key = tuple(client.values())
        with self._lock:
            if (
                key in self._clients_seen
                or len(self._clients_seen) >= _MAX_CLIENTS_SEEN
            ):
                return
            self._clients_seen.add(key)
        self.capture("client_connected", **client)

    def _activate(self, time_to_ready_ms: int | None) -> None:
        assert self._context is not None
        try:
            state = _read_state()
            robot_id = state.get("robot_id")
            if not isinstance(robot_id, str) or not robot_id:
                robot_id = str(uuid.uuid4())
            envelope = _envelope(**self._context)
            previous_version = state.get("last_version")
            previous_exit_clean = state.get("running") is not True
            _write_state(
                {
                    **state,
                    "robot_id": robot_id,
                    "last_version": envelope["daemon_version"],
                    "running": True,
                }
            )
        except Exception as e:  # noqa: BLE001 - telemetry must never break the daemon
            logger.warning("Telemetry disabled for this run: %s", e)
            return

        with self._lock:
            # A flush still in flight from a previous identity may have requeued.
            self._queue.clear()
            self._clients_seen.clear()
            self._robot_id = robot_id
            self._envelope = envelope
        started: dict[str, Any] = {"previous_exit_clean": previous_exit_clean}
        if time_to_ready_ms is not None:
            started["time_to_ready_ms"] = time_to_ready_ms
        self.capture("daemon_started", **started)
        if (
            isinstance(previous_version, str)
            and previous_version != envelope["daemon_version"]
        ):
            self.capture(
                "update_result",
                from_version=previous_version,
                to_version=envelope["daemon_version"],
                success=True,
            )

        # One stop event per thread: a previous thread still stuck in a send
        # must not be revived by a quick opt-out / opt-in.
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(self._stop,), name="telemetry", daemon=True
        )
        self._thread.start()

    def _halt(self, flush: bool) -> None:
        with self._lock:
            self._envelope = None
            self._robot_id = None
            if not flush:
                self._queue.clear()
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self, stop: threading.Event) -> None:
        next_heartbeat = time.monotonic() + HEARTBEAT_INTERVAL_S
        started = time.monotonic()
        while not stop.is_set():
            self._wake.wait(FLUSH_INTERVAL_S)
            self._wake.clear()
            if time.monotonic() >= next_heartbeat:
                self.capture(
                    "daily_heartbeat", uptime_s=int(time.monotonic() - started)
                )
                next_heartbeat += HEARTBEAT_INTERVAL_S
            self._flush(timeout=10.0)
        self._flush(timeout=2.0)

    def _flush(self, timeout: float) -> None:
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()
        if not batch:
            return
        try:
            response = requests.post(
                POSTHOG_URL,
                json={"api_key": POSTHOG_KEY, "batch": batch},
                timeout=timeout,
            )
            retry = response.status_code == 429 or response.status_code >= 500
            if response.ok:
                logger.info("Telemetry sent: %s", ", ".join(e["event"] for e in batch))
            elif not retry:
                logger.warning("Telemetry rejected (HTTP %d)", response.status_code)
        except requests.RequestException as e:
            logger.debug("Telemetry flush failed: %s", e)
            retry = True
        if retry:
            # Offline or throttled: retry at the next tick, newest dropped first.
            # Dropped if the user opted out (or got a new id) meanwhile.
            with self._lock:
                if batch[0]["distinct_id"] == self._robot_id:
                    self._queue.extendleft(reversed(batch))


telemetry = Telemetry()
