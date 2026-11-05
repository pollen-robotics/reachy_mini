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
USAGE_TICK_S = 3600.0
_MAX_QUEUE = 500
_MAX_CLIENTS_SEEN = 50
_MAX_SESSIONS = 50
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})

CLIENT_HEADER = "X-Reachy-Client"
CLIENT_TYPES = frozenset({"desktop", "mobile", "web", "python_sdk"})
CLIENT_OSES = frozenset({"macos", "windows", "linux", "ios", "android"})
_CLIENT_RE = re.compile(
    r"^([a-z_]{1,16})/([0-9A-Za-z.+-]{1,32})(?: \(([a-z]{1,16})\))?$"
)
_HF_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}$", re.ASCII)
_MOVE_NAME_RE = re.compile(r"^[\w.-]{1,64}$", re.ASCII)
# shortcut: only Pollen's own move libraries are named, other datasets report
# "other"; upgrade to a public check if community libraries matter.
_NAMED_MOVE_OWNERS = ("pollen-robotics/",)


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
    client = {
        "client_type": client_type if client_type in CLIENT_TYPES else "other",
        "client_version": client_version,
    }
    if client_os is not None:  # browsers don't say: leave it out, not "other"
        client["client_os"] = client_os if client_os in CLIENT_OSES else "other"
    if client["client_type"] == "other":
        # Already limited to 16 lowercase letters by the regex.
        client["client_type_raw"] = client_type
    return client


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
        # Serialises set_enabled between the HTTP route and the data channel.
        self._toggle_lock = threading.Lock()
        self._context: dict[str, Any] | None = None
        self._envelope: dict[str, Any] | None = None
        self._robot_id: str | None = None
        self._clients_seen: set[tuple[str, ...]] = set()
        # Who is using the robot right now: key -> (kind, app id, start time),
        # and everything seen since the last usage tick.
        self._sessions: dict[str, tuple[str, str | None, float]] = {}
        self._window: dict[str, tuple[str, str | None]] = {}
        # Seconds with at least one session open since the last tick.
        self._clock = time.monotonic
        self._busy_since: float | None = None
        self._busy_s = 0.0

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
        self._usage_tick()  # the use since the last tick, sent with the final flush
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
        with self._toggle_lock:
            startup_app_config.set_telemetry_enabled(enabled)
            if enabled and not self.active and self._context and not disabled_by_env():
                self._activate(time_to_ready_ms=None)
            elif not enabled:
                if self.active:
                    # No join: the thread has its own stop event and its
                    # requeue is dropped once the robot id is gone.
                    self._halt(flush=False, wait=False)
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

    def session_begin(self, key: str, kind: str, app_id: str | None = None) -> None:
        """Count *key* as using the robot until ``session_end``."""
        now = self._clock()
        with self._lock:
            if key not in self._sessions and len(self._sessions) >= _MAX_SESSIONS:
                return
            if not self._sessions:
                self._busy_since = now
            self._sessions[key] = (kind, app_id, now)
            if self._envelope is not None:  # nothing piles up while opted out
                self._window[key] = (kind, app_id)

    def session_end(self, key: str) -> tuple[str, str | None, float] | None:
        """Stop counting *key*; return its session, or None if unknown."""
        with self._lock:
            session = self._sessions.pop(key, None)
            ended_last = session is not None and not self._sessions
            if ended_last and self._busy_since is not None:
                self._busy_s += self._clock() - self._busy_since
                self._busy_since = None
            return session

    def note_hello(self, peer_id: str, client: str, app: str | None) -> None:
        """Track a data-channel client and the JS app it runs, if any."""
        parsed = parse_client_header(client)
        if parsed is None:
            return
        self.note_client(client)
        if app is not None and (app == "private" or not _HF_ID_RE.match(app)):
            app = "private"  # the SDK only names Spaces it found public
        key = f"peer:{peer_id}"
        with self._lock:
            session = self._sessions.get(key)
        if session is not None and session[:2] == (parsed["client_type"], app):
            return  # same client again (e.g. a reconnect): keep its session
        # Relayed sessions share one peer id, but the relay runs one at a
        # time: a different hello is a new session replacing the old one.
        self.peer_closed(peer_id)
        self.session_begin(key, parsed["client_type"], app)
        if app is not None:
            self.capture("js_app_started", app_id=app)

    def peer_closed(self, peer_id: str) -> None:
        """End a data-channel session, reporting the JS app's run time."""
        session = self.session_end(f"peer:{peer_id}")
        if session is not None and session[1] is not None:
            self.capture(
                "js_app_stopped",
                app_id=session[1],
                duration_s=int(self._clock() - session[2]),
            )

    def note_move(self, dataset: str, move: str) -> None:
        """Report a recorded move, named only for Pollen's own libraries."""
        if dataset.startswith(_NAMED_MOVE_OWNERS) and _MOVE_NAME_RE.match(move):
            self.capture("move_played", dataset=dataset, move=move)
        else:
            self.capture("move_played", dataset="other")

    def _maybe_tick(self, next_tick: float) -> float:
        """Tick when use stops, or hourly while it goes on; return the next deadline.

        Ticking only on the hour would never send a run shorter than an hour,
        nor the last partial hour, since a daemon often exits uncleanly.
        """
        now = self._clock()
        with self._lock:
            idle = not self._sessions
            used = bool(self._window)
        if not used:
            return now + USAGE_TICK_S  # the hour starts with the next use
        if idle or now >= next_tick:
            self._usage_tick()
            # From now, not +=: no burst of ticks after a suspend.
            return now + USAGE_TICK_S
        return next_tick

    def _usage_tick(self) -> None:
        now = self._clock()
        with self._lock:
            window = self._window
            self._window = {
                k: (kind, app) for k, (kind, app, _) in self._sessions.items()
            }
            active_s = self._busy_s
            if self._busy_since is not None:
                active_s += now - self._busy_since
                self._busy_since = now
            self._busy_s = 0.0
        if window and active_s > 0:  # a session ending right at the last tick
            kinds = {kind for kind, _ in window.values()}
            if "python_app" in kinds:
                # The app's own SDK connection: not a second client.
                kinds.discard("python_sdk")
            self.capture(
                "usage_tick",
                interval_s=int(USAGE_TICK_S),
                active_s=round(active_s),
                kinds=sorted(kinds),
                apps=sorted({app for _, app in window.values() if app}),
            )

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
            # Report only what is in use from now on, not what ran while off.
            self._window = {
                k: (kind, app) for k, (kind, app, _) in self._sessions.items()
            }
            self._busy_since = self._clock() if self._sessions else None
            self._busy_s = 0.0
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

    def _halt(self, flush: bool, wait: bool = True) -> None:
        with self._lock:
            self._envelope = None
            self._robot_id = None
            self._window = {}
            if not flush:
                self._queue.clear()
        self._stop.set()
        self._wake.set()
        if self._thread is not None and wait:
            self._thread.join(timeout=3.0)
        self._thread = None

    def _run(self, stop: threading.Event) -> None:
        next_heartbeat = time.monotonic() + HEARTBEAT_INTERVAL_S
        next_tick = self._clock() + USAGE_TICK_S
        started = time.monotonic()
        while not stop.is_set():
            self._wake.wait(FLUSH_INTERVAL_S)
            self._wake.clear()
            now = time.monotonic()
            if now >= next_heartbeat:
                self.capture("daily_heartbeat", uptime_s=int(now - started))
                next_heartbeat += HEARTBEAT_INTERVAL_S
            next_tick = self._maybe_tick(next_tick)
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
