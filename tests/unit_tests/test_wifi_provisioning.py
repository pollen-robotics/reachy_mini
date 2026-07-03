"""Unit tests for QR-based WiFi provisioning (parser + orchestration).

Pure and offline: the provisioner is driven with fake camera frames, a fake
QR decoder, and a fake connect function. No camera, no nmcli, no daemon.
"""

import time

from reachy_mini.daemon.app.services.wifi_provisioning import (
    ProvisioningState,
    QrWifiProvisioner,
    parse_wifi_qr,
)

# ---------------------------------------------------------------------------
# WIFI: QR payload parser (the de-facto standard used by Android/iOS)
# ---------------------------------------------------------------------------


def test_parses_standard_payload():
    creds = parse_wifi_qr("WIFI:T:WPA;S:mynetwork;P:mypassword;;")
    assert creds is not None
    assert creds.ssid == "mynetwork"
    assert creds.password == "mypassword"
    assert creds.security == "WPA"
    assert creds.hidden is False


def test_parses_any_field_order_and_hidden():
    creds = parse_wifi_qr("WIFI:S:net;H:true;P:pw;T:WPA;;")
    assert creds is not None
    assert creds.ssid == "net"
    assert creds.hidden is True


def test_parses_escaped_characters():
    # \; \: \, \\ are escaped in SSIDs and passwords
    creds = parse_wifi_qr(r"WIFI:T:WPA;S:my\;we\:ird\,ssid;P:p\\ss;;")
    assert creds is not None
    assert creds.ssid == "my;we:ird,ssid"
    assert creds.password == r"p\ss"


def test_parses_open_network():
    creds = parse_wifi_qr("WIFI:T:nopass;S:open-net;;")
    assert creds is not None
    assert creds.ssid == "open-net"
    assert creds.password is None
    assert creds.security == "nopass"


def test_rejects_non_wifi_payloads():
    assert parse_wifi_qr("http://example.com") is None
    assert parse_wifi_qr("") is None
    assert parse_wifi_qr("WIFI:T:WPA;P:pw;;") is None  # no SSID


# ---------------------------------------------------------------------------
# Provisioner orchestration, with injected fakes
# ---------------------------------------------------------------------------


class FakeCamera:
    def __init__(self):
        self.closed = False

    def read(self):
        return "frame"  # any non-None object

    def close(self):
        self.closed = True


def make_provisioner(
    payloads=None,
    connected_after=0.0,
    camera=None,
    decoder_available=True,
    scan_timeout_s=0.5,
    connect_timeout_s=0.5,
):
    """Provisioner wired to fakes. `payloads` = QR text per frame (cycled)."""
    camera = camera if camera is not None else FakeCamera()
    seen = {"n": 0, "connect": None, "sounds": [], "t_connect": None}

    def camera_factory():
        return camera

    def qr_decode(frame):
        if payloads is None:
            return None
        p = payloads[min(seen["n"], len(payloads) - 1)]
        seen["n"] += 1
        return p

    def connect(ssid, password):
        seen["connect"] = (ssid, password)
        seen["t_connect"] = time.monotonic()

    def wifi_connected():
        t = seen["t_connect"]
        return t is not None and time.monotonic() - t >= connected_after

    prov = QrWifiProvisioner(
        camera_factory=camera_factory,
        qr_decode=qr_decode if decoder_available else None,
        connect=connect,
        wifi_connected=wifi_connected,
        play_sound=lambda name: seen["sounds"].append(name),
        scan_timeout_s=scan_timeout_s,
        connect_timeout_s=connect_timeout_s,
        scan_period_s=0.01,
    )
    return prov, seen


def wait_done(prov, timeout=5.0):
    t0 = time.monotonic()
    while prov.status().state in (
        ProvisioningState.SCANNING,
        ProvisioningState.CONNECTING,
    ):
        assert time.monotonic() - t0 < timeout, "provisioner did not finish"
        time.sleep(0.01)


def test_full_success_path():
    prov, seen = make_provisioner(payloads=[None, None, "WIFI:T:WPA;S:net;P:pw;;"])
    assert prov.start()
    wait_done(prov)
    assert prov.status().state == ProvisioningState.SUCCESS
    assert prov.status().ssid == "net"
    assert seen["connect"] == ("net", "pw")
    assert "wifi_scanning.wav" in seen["sounds"]
    assert "handshake_success.wav" in seen["sounds"]


def test_ignores_non_wifi_qr_and_keeps_scanning():
    prov, seen = make_provisioner(
        payloads=["http://nope", "WIFI:T:WPA;S:net;P:pw;;"]
    )
    prov.start()
    wait_done(prov)
    assert prov.status().state == ProvisioningState.SUCCESS


def test_times_out_when_no_qr():
    prov, seen = make_provisioner(payloads=None, scan_timeout_s=0.1)
    prov.start()
    wait_done(prov)
    assert prov.status().state == ProvisioningState.FAILED
    assert "handshake_aborted.wav" in seen["sounds"]


def test_fails_when_connection_never_comes_up():
    prov, seen = make_provisioner(
        payloads=["WIFI:T:WPA;S:net;P:pw;;"],
        connected_after=99.0,
        connect_timeout_s=0.1,
    )
    prov.start()
    wait_done(prov)
    assert prov.status().state == ProvisioningState.FAILED


def test_unavailable_without_decoder():
    prov, seen = make_provisioner(decoder_available=False)
    assert not prov.start()
    assert prov.status().state == ProvisioningState.UNAVAILABLE


def test_no_double_start_and_camera_closed():
    cam = FakeCamera()
    prov, seen = make_provisioner(
        payloads=["WIFI:T:WPA;S:net;P:pw;;"], camera=cam
    )
    assert prov.start()
    prov.start()  # second start while running: refused, no crash
    wait_done(prov)
    assert cam.closed
    # after finishing, a new run may start again
    assert prov.start()
    wait_done(prov)
