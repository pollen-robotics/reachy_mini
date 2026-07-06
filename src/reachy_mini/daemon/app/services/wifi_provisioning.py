"""QR-based WiFi provisioning for the Wireless Reachy Mini.

The robot shows the user its camera, the user shows it a WiFi QR code (the
standard `WIFI:T:WPA;S:ssid;P:password;;` payload that Android and iOS
generate), and the robot connects itself. Fully offline until the connect
step, so it works on a freshly built robot with no network.

The flow (QrWifiProvisioner.start(), runs in its own thread):

    (intro)      plays the narrated wifi_setup_intro.wav immediately and
                 runs the injected `prepare` hook (torque on + goto the
                 base pose so the camera looks forward, 1.5 s).
    SCANNING     once the narration is over (intro_wait_s), grab camera
                 frames (~4 Hz) and look for a WiFi QR code, up to
                 scan_timeout_s (one minute).
    CONNECTING   credentials found: hand them to the connect function
                 (the same nmcli path as the /wifi/connect endpoint) and
                 wait for the WiFi to come up.
    SUCCESS      connected: plays handshake_success.wav.
    FAILED       no QR in time, or the connection never came up: plays
                 handshake_aborted.wav.
    UNAVAILABLE  no QR decoder (opencv not installed) or no camera.

    Every exit path runs the injected `finish` hook (goto sleep + torque
    off, which also re-arms the secret handshake).

Every dependency (camera, QR decoder, connect, status, sounds) is injected,
so the whole flow is unit-tested offline (test_wifi_provisioning.py); the
production wiring lives in the wifi_provision router. The camera frames come
from the media server's existing IPC branch (GStreamerCamera), the same
mechanism local apps use, so nothing in the media pipeline changes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)

SOUND_INTRO = "wifi_setup_intro.wav"  # narrated voice explaining the flow
SOUND_SUCCESS = "handshake_success.wav"
SOUND_FAILED = "handshake_aborted.wav"


@dataclass(frozen=True)
class WifiQrCredentials:
    """Credentials decoded from a WiFi QR code."""

    ssid: str
    password: Optional[str]  # None for open networks
    security: str  # "WPA", "WEP", "nopass", ...
    hidden: bool


def parse_wifi_qr(payload: str) -> Optional[WifiQrCredentials]:
    r"""Parse the de-facto standard WiFi QR payload, None if not one.

    Format: `WIFI:T:WPA;S:my ssid;P:my pass;H:true;;` with `\\`-escaping of
    the special characters `\\ ; , : "` inside values. Field order is free.
    """
    if not payload or not payload.startswith("WIFI:"):
        return None

    fields: dict[str, str] = {}
    key: Optional[str] = None
    value: list[str] = []
    i = len("WIFI:")
    n = len(payload)
    while i < n:
        c = payload[i]
        if key is None:
            colon = payload.find(":", i)
            if colon == -1:
                break
            key = payload[i:colon].strip().upper()
            value = []
            i = colon + 1
        elif c == "\\" and i + 1 < n:
            value.append(payload[i + 1])
            i += 2
        elif c == ";":
            fields[key] = "".join(value)
            key = None
            i += 1
        else:
            value.append(c)
            i += 1

    ssid = fields.get("S", "")
    if not ssid:
        return None
    security = fields.get("T", "nopass") or "nopass"
    password = fields.get("P") or None
    if security.lower() == "nopass":
        password = None
    hidden = fields.get("H", "").lower() == "true"
    return WifiQrCredentials(
        ssid=ssid, password=password, security=security, hidden=hidden
    )


class ProvisioningState(str, Enum):
    """Lifecycle of a provisioning run."""

    IDLE = "idle"
    SCANNING = "scanning"
    CONNECTING = "connecting"
    SUCCESS = "success"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass
class ProvisioningStatus:
    """What the provisioner is doing right now (safe to expose over REST)."""

    state: ProvisioningState = ProvisioningState.IDLE
    detail: str = ""
    ssid: Optional[str] = None


class QrWifiProvisioner:
    """Camera -> QR -> nmcli orchestrator. All dependencies injected.

    camera_factory   () -> object with read() -> frame|None and close();
                     called at scan start, closed when the run ends.
    qr_decode        (frame) -> decoded text or None; None-the-callable
                     means "no decoder available" (opencv not installed).
    connect          (ssid, password|None) -> None; the /wifi/connect
                     entry point (returns immediately, work in its thread).
    wifi_status      () -> (mode, connected_network) with mode one of
                     "busy" / "wlan" / "hotspot" / "disconnected", i.e.
                     what GET /wifi/status reports. Polled to confirm the
                     outcome with the SAME semantics as the desktop app's
                     onboarding: success = wlan on the target ssid,
                     hotspot after busy = the daemon reverted (failed).
    play_sound       (asset_name) -> None.
    prepare          optional; runs first, before the camera opens. The
                     daemon wires "torque on + goto the base pose" here so
                     the camera looks forward instead of at the table.
    finish           optional; runs on EVERY exit path (success, failure,
                     no camera). The daemon wires "goto sleep + torque
                     off" here, which also re-arms the handshake.
    intro_wait_s     scanning (and its one-minute clock) only starts this
                     long after the intro narration begins, so fumbling
                     for a QR code during the explanation costs nothing.
    """

    def __init__(
        self,
        camera_factory: Callable[[], object],
        qr_decode: Optional[Callable[[object], Optional[str]]],
        connect: Callable[[str, Optional[str]], None],
        wifi_status: Callable[[], tuple[str, Optional[str]]],
        play_sound: Callable[[str], None],
        prepare: Optional[Callable[[], None]] = None,
        finish: Optional[Callable[[], None]] = None,
        scan_timeout_s: float = 60.0,
        connect_timeout_s: float = 45.0,
        scan_period_s: float = 0.25,
        intro_wait_s: float = 0.0,
    ) -> None:
        """Store the injected dependencies (see class docstring)."""
        self._camera_factory = camera_factory
        self._qr_decode = qr_decode
        self._connect = connect
        self._wifi_status = wifi_status
        self._play_sound = play_sound
        self._prepare = prepare
        self._finish = finish
        self.scan_timeout_s = scan_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.scan_period_s = scan_period_s
        self.intro_wait_s = intro_wait_s
        self._status = ProvisioningStatus()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def status(self) -> ProvisioningStatus:
        """Return the current / last run status."""
        return self._status

    def start(self) -> bool:
        """Start a provisioning run. False if running or unavailable."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self._qr_decode is None:
                self._status = ProvisioningStatus(
                    ProvisioningState.UNAVAILABLE,
                    "no QR decoder available (opencv not installed)",
                )
                return False
            self._status = ProvisioningStatus(ProvisioningState.SCANNING)
            self._thread = threading.Thread(
                target=self._run, name="wifi-qr-provisioning", daemon=True
            )
            self._thread.start()
            return True

    # ------------------------------------------------------------------
    def _run(self) -> None:
        # The narration gives instant feedback; the robot rises to the base
        # pose while it plays; scanning starts once the narration is over.
        started_at = time.monotonic()
        self._safe_sound(SOUND_INTRO)
        self._safe_hook(self._prepare, "prepare")
        try:
            self._run_provisioning(started_at)
        finally:
            # Back to sleep + torque off on every exit, re-arming the
            # handshake for the next attempt.
            self._safe_hook(self._finish, "finish")

    def _run_provisioning(self, started_at: float) -> None:
        try:
            camera = self._camera_factory()
        except Exception as e:
            logger.exception("QR provisioning: camera unavailable")
            self._status = ProvisioningStatus(
                ProvisioningState.UNAVAILABLE, f"camera unavailable: {e}"
            )
            return
        try:
            remaining_intro = self.intro_wait_s - (time.monotonic() - started_at)
            if remaining_intro > 0:
                time.sleep(remaining_intro)
            creds = self._scan(camera)
        finally:
            try:
                camera.close()
            except Exception:
                logger.exception("QR provisioning: camera close failed")

        if creds is None:
            self._status = ProvisioningStatus(
                ProvisioningState.FAILED, "no WiFi QR code seen in time"
            )
            self._safe_sound(SOUND_FAILED)
            return

        logger.info(f"QR provisioning: connecting to '{creds.ssid}'")
        self._status = ProvisioningStatus(ProvisioningState.CONNECTING, ssid=creds.ssid)
        try:
            self._connect(creds.ssid, creds.password)
        except Exception as e:
            logger.exception("QR provisioning: connect call failed")
            self._status = ProvisioningStatus(
                ProvisioningState.FAILED, f"connect failed: {e}", ssid=creds.ssid
            )
            self._safe_sound(SOUND_FAILED)
            return

        # Confirm the outcome exactly like the desktop app onboarding does:
        # poll the wifi status; "busy" = still working; "wlan" on the target
        # ssid = success; "hotspot" once the work started = the daemon gave
        # up and reverted to its AP (e.g. wrong password).
        deadline = time.monotonic() + self.connect_timeout_s
        saw_busy = False
        while time.monotonic() < deadline:
            try:
                mode, network = self._wifi_status()
            except Exception:
                logger.exception("QR provisioning: wifi status poll failed")
                mode, network = ("disconnected", None)
            if mode == "busy":
                saw_busy = True
            elif mode == "wlan" and network == creds.ssid:
                self._status = ProvisioningStatus(
                    ProvisioningState.SUCCESS, ssid=creds.ssid
                )
                self._safe_sound(SOUND_SUCCESS)
                return
            elif mode == "hotspot" and saw_busy:
                self._status = ProvisioningStatus(
                    ProvisioningState.FAILED,
                    "daemon reverted to hotspot (wrong password?)",
                    ssid=creds.ssid,
                )
                self._safe_sound(SOUND_FAILED)
                return
            time.sleep(self.scan_period_s)

        self._status = ProvisioningStatus(
            ProvisioningState.FAILED,
            "connection did not come up in time",
            ssid=creds.ssid,
        )
        self._safe_sound(SOUND_FAILED)

    def _scan(self, camera) -> Optional[WifiQrCredentials]:
        assert self._qr_decode is not None
        deadline = time.monotonic() + self.scan_timeout_s
        while time.monotonic() < deadline:
            frame = None
            try:
                frame = camera.read()
            except Exception:
                logger.exception("QR provisioning: camera read failed")
            if frame is not None:
                try:
                    text = self._qr_decode(frame)
                except Exception:
                    logger.exception("QR provisioning: QR decode failed")
                    text = None
                if text:
                    creds = parse_wifi_qr(text)
                    if creds is not None:
                        return creds
            time.sleep(self.scan_period_s)
        return None

    def _safe_sound(self, name: str) -> None:
        try:
            self._play_sound(name)
        except Exception:
            logger.exception("QR provisioning: sound failed")

    def _safe_hook(self, hook: Optional[Callable[[], None]], name: str) -> None:
        if hook is None:
            return
        try:
            hook()
        except Exception:
            logger.exception(f"QR provisioning: {name} hook failed")


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------

_shared: Optional[QrWifiProvisioner] = None
_shared_lock = threading.Lock()


def get_shared_provisioner(
    play_sound: Callable[[str], None],
    camera_specs: object = None,
    backend: object = None,
) -> QrWifiProvisioner:
    """Return the daemon-wide provisioner (lazy singleton, thread-safe)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = build_default_provisioner(play_sound, camera_specs, backend)
        return _shared


def _intro_duration_s() -> float:
    """Length of the narration wav, so scanning starts after it ends."""
    import contextlib
    import wave
    from importlib import resources

    try:
        asset = resources.files("reachy_mini") / "assets" / SOUND_INTRO
        with resources.as_file(asset) as path, contextlib.closing(
            wave.open(str(path), "rb")
        ) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        logger.exception("QR provisioning: could not read intro duration")
        return 0.0


def build_default_provisioner(
    play_sound: Callable[[str], None],
    camera_specs: object = None,
    backend: object = None,
) -> QrWifiProvisioner:
    """Wire the provisioner to the real robot.

    Camera: the media server's IPC branch via GStreamerCamera (the same
    mechanism local apps use, nothing in the media pipeline changes).
    QR decoding: cv2.QRCodeDetector, imported lazily; when opencv is not
    installed the provisioner reports UNAVAILABLE instead of crashing.
    Connect + confirm: literally the endpoints the desktop app onboarding
    drives (POST /wifi/connect then polling GET /wifi/status), so behavior
    including the revert-to-hotspot fallback stays identical to the
    existing, battle-tested flow.
    Robot motion: prepare = torque on + goto the base pose (1.5 s) so the
    camera looks forward; finish = goto sleep + torque off on every exit,
    which also re-arms the secret handshake.
    """

    def camera_factory() -> object:
        from reachy_mini.media.camera_gstreamer import GStreamerCamera

        camera = GStreamerCamera(camera_specs=camera_specs)  # type: ignore[arg-type]
        camera.open()
        return camera

    qr_decode: Optional[Callable[[object], Optional[str]]] = None
    try:
        import cv2

        detector = cv2.QRCodeDetector()

        def qr_decode(frame: object) -> Optional[str]:
            text, _points, _raw = detector.detectAndDecode(frame)
            return text or None

    except ImportError:
        logger.warning("QR provisioning unavailable: opencv is not installed")

    def connect(ssid: str, password: Optional[str]) -> None:
        # The /wifi/connect route function itself: busy-lock handling,
        # worker thread, error capture, revert-to-hotspot fallback.
        from ..routers.wifi_config import connect_to_wifi_network

        connect_to_wifi_network(ssid=ssid, password=password or "")

    def wifi_status() -> tuple[str, Optional[str]]:
        # The /wifi/status route function, reduced to what the poll needs.
        from ..routers.wifi_config import get_wifi_status

        status = get_wifi_status()
        return status.mode.value, status.connected_network

    prepare: Optional[Callable[[], None]] = None
    finish: Optional[Callable[[], None]] = None
    if backend is not None and hasattr(backend, "enable_motors"):

        def prepare() -> None:
            import asyncio

            backend.enable_motors()  # pins the present pose: no snap
            asyncio.run(
                backend.goto_target(
                    head=backend.INIT_HEAD_POSE,
                    antennas=backend.INIT_ANTENNAS_JOINT_POSITIONS,
                    duration=1.5,
                )
            )

        def finish() -> None:
            import asyncio

            asyncio.run(backend.goto_sleep())
            backend.disable_motors()

    return QrWifiProvisioner(
        camera_factory=camera_factory,
        qr_decode=qr_decode,
        connect=connect,
        wifi_status=wifi_status,
        play_sound=play_sound,
        prepare=prepare,
        finish=finish,
        intro_wait_s=_intro_duration_s(),
    )
