"""QR-based WiFi provisioning endpoints (wireless version only).

Show the robot a standard WiFi QR code (`WIFI:T:WPA;S:ssid;P:pass;;`, the
format Android and iOS generate) and it connects itself. The scanning /
connecting flow lives in services/wifi_provisioning.py; this router only
starts it and reports its status.

    POST /wifi/provision_qr/start   -> {"started": bool, "state": ...}
    GET  /wifi/provision_qr/status  -> {"state", "detail", "ssid"}
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ...daemon import Daemon
from ..dependencies import get_daemon
from ..services.wifi_provisioning import get_shared_provisioner

router = APIRouter(prefix="/wifi/provision_qr")
logger = logging.getLogger(__name__)


def _provisioner(daemon: Daemon):  # type: ignore[no-untyped-def]
    backend = daemon.backend
    if backend is None or not backend.ready.is_set():
        raise HTTPException(status_code=503, detail="Backend not running")
    media_server = getattr(daemon, "_media_server", None)
    camera_specs = getattr(media_server, "camera_specs", None)
    return get_shared_provisioner(backend.play_sound, camera_specs, backend)


@router.post("/start")
def start_qr_provisioning(daemon: Daemon = Depends(get_daemon)) -> dict:
    """Start scanning the camera for a WiFi QR code."""
    provisioner = _provisioner(daemon)
    started = provisioner.start()
    status = provisioner.status()
    return {"started": started, "state": status.state, "detail": status.detail}


@router.get("/status")
def qr_provisioning_status(daemon: Daemon = Depends(get_daemon)) -> dict:
    """Status of the current / last QR provisioning run."""
    status = _provisioner(daemon).status()
    return {"state": status.state, "detail": status.detail, "ssid": status.ssid}
