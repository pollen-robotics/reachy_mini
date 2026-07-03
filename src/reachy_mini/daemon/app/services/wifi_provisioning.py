"""QR-based WiFi provisioning for the Wireless Reachy Mini.

The robot shows the user its camera, the user shows it a WiFi QR code (the
standard `WIFI:T:WPA;S:ssid;P:password;;` payload that Android and iOS
generate), and the robot connects itself. Fully offline until the connect
step, so it works on a freshly built robot with no network.

The flow (QrWifiProvisioner.start(), runs in its own thread):

    SCANNING     grab camera frames (~4 Hz) and look for a WiFi QR code,
                 up to scan_timeout_s. Plays wifi_scanning.wav at start.
    CONNECTING   credentials found: hand them to the connect function
                 (the same nmcli path as the /wifi/connect endpoint) and
                 wait for the WiFi to come up.
    SUCCESS      connected: plays handshake_success.wav.
    FAILED       no QR in time, or the connection never came up: plays
                 handshake_aborted.wav.
    UNAVAILABLE  no QR decoder (opencv not installed) or no camera.

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

SOUND_SCANNING = "wifi_scanning.wav"
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
    connect          (ssid, password|None) -> None; must block or spawn its
                     own worker; errors surface through wifi_connected
                     staying False.
    wifi_connected   () -> bool; polled to confirm the connection came up.
    play_sound       (asset_name) -> None.
    """

    def __init__(
        self,
        camera_factory: Callable[[], object],
        qr_decode: Optional[Callable[[object], Optional[str]]],
        connect: Callable[[str, Optional[str]], None],
        wifi_connected: Callable[[], bool],
        play_sound: Callable[[str], None],
        scan_timeout_s: float = 90.0,
        connect_timeout_s: float = 45.0,
        scan_period_s: float = 0.25,
    ) -> None:
        """Store the injected dependencies (see class docstring)."""
        self._camera_factory = camera_factory
        self._qr_decode = qr_decode
        self._connect = connect
        self._wifi_connected = wifi_connected
        self._play_sound = play_sound
        self.scan_timeout_s = scan_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.scan_period_s = scan_period_s
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
        try:
            camera = self._camera_factory()
        except Exception as e:
            logger.exception("QR provisioning: camera unavailable")
            self._status = ProvisioningStatus(
                ProvisioningState.UNAVAILABLE, f"camera unavailable: {e}"
            )
            return
        try:
            self._safe_sound(SOUND_SCANNING)
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

        deadline = time.monotonic() + self.connect_timeout_s
        while time.monotonic() < deadline:
            if self._wifi_connected():
                self._status = ProvisioningStatus(
                    ProvisioningState.SUCCESS, ssid=creds.ssid
                )
                self._safe_sound(SOUND_SUCCESS)
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


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------

_shared: Optional[QrWifiProvisioner] = None
_shared_lock = threading.Lock()


def get_shared_provisioner(
    play_sound: Callable[[str], None], camera_specs: object = None
) -> QrWifiProvisioner:
    """Return the daemon-wide provisioner (lazy singleton, thread-safe)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = build_default_provisioner(play_sound, camera_specs)
        return _shared


def build_default_provisioner(
    play_sound: Callable[[str], None], camera_specs: object = None
) -> QrWifiProvisioner:
    """Wire the provisioner to the real robot.

    Camera: the media server's IPC branch via GStreamerCamera (the same
    mechanism local apps use, nothing in the media pipeline changes).
    QR decoding: cv2.QRCodeDetector, imported lazily; when opencv is not
    installed the provisioner reports UNAVAILABLE instead of crashing.
    Connect: the exact nmcli path of the /wifi/connect endpoint, including
    its revert-to-hotspot fallback, under the same busy lock.
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
        from ..routers import wifi_config

        with wifi_config.busy_lock:
            try:
                wifi_config.setup_wifi_connection(
                    name=ssid, ssid=ssid, password=password or ""
                )
            except Exception:
                logger.exception(
                    f"QR provisioning: connect to '{ssid}' failed, reverting to hotspot"
                )
                wifi_config.remove_connection(name=ssid)
                wifi_config.setup_wifi_connection(
                    name="Hotspot",
                    ssid=wifi_config.HOTSPOT_SSID,
                    password=wifi_config.HOTSPOT_PASSWORD,
                    is_hotspot=True,
                )
                raise

    def wifi_connected() -> bool:
        from ..routers import wifi_config

        return wifi_config.get_current_wifi_mode() == wifi_config.WifiMode.WLAN

    return QrWifiProvisioner(
        camera_factory=camera_factory,
        qr_decode=qr_decode,
        connect=connect,
        wifi_connected=wifi_connected,
        play_sound=play_sound,
    )
