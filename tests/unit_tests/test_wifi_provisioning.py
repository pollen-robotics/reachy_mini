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
    status_after_connect=None,
    camera=None,
    decoder_available=True,
    scan_timeout_s=0.5,
    connect_timeout_s=0.5,
    intro_wait_s=0.0,
):
    """Provisioner wired to fakes.

    `payloads` = QR text per frame (last one repeats).
    `status_after_connect` = sequence of (mode, connected_network) returned
    by the wifi status poll once connect() was called (last one repeats),
    mirroring what the daemon's /wifi/status reports and what the desktop
    app polls during onboarding. Before connect: ("disconnected", None).
    """
    camera = camera if camera is not None else FakeCamera()
    if status_after_connect is None:
        status_after_connect = [("busy", None), ("wlan", "net")]
    seen = {
        "n": 0,
        "n_status": 0,
        "connect": None,
        "sounds": [],
        "events": [],
        "first_decode_t": None,
        "started_t": None,
        "frames_saved": [],
    }

    def camera_factory():
        return camera

    def qr_decode(frame):
        if seen["first_decode_t"] is None:
            seen["first_decode_t"] = time.monotonic()
        if payloads is None:
            return None
        p = payloads[min(seen["n"], len(payloads) - 1)]
        seen["n"] += 1
        return p

    def connect(ssid, password):
        seen["connect"] = (ssid, password)

    def wifi_status():
        if seen["connect"] is None:
            return ("disconnected", None)
        s = status_after_connect[min(seen["n_status"], len(status_after_connect) - 1)]
        seen["n_status"] += 1
        return s

    def play_sound(name):
        seen["sounds"].append(name)
        seen["events"].append(f"sound:{name}")

    prov = QrWifiProvisioner(
        camera_factory=camera_factory,
        qr_decode=qr_decode if decoder_available else None,
        connect=connect,
        wifi_status=wifi_status,
        play_sound=play_sound,
        prepare=lambda: seen["events"].append("prepare"),
        finish=lambda: seen["events"].append("finish"),
        save_frame=lambda i, frame: seen["frames_saved"].append(i),
        scan_timeout_s=scan_timeout_s,
        connect_timeout_s=connect_timeout_s,
        scan_period_s=0.01,
        intro_wait_s=intro_wait_s,
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
    assert "wifi_setup_intro.wav" in seen["sounds"]
    assert "wifi_qr_detected.wav" in seen["sounds"]
    assert "wifi_connect_success.wav" in seen["sounds"]
    # every scanned frame was handed to the frame sink (diag snapshots)
    assert seen["frames_saved"] == list(range(len(seen["frames_saved"])))
    assert len(seen["frames_saved"]) >= 1


def test_prepare_and_finish_wrap_the_run():
    """The robot wakes before scanning and returns to sleep at the end."""
    prov, seen = make_provisioner(payloads=["WIFI:T:WPA;S:net;P:pw;;"])
    prov.start()
    wait_done(prov)
    events = seen["events"]
    # narration first (instant feedback), the robot rises while it plays,
    # and finish comes after the final outcome sound
    assert events[0] == "sound:wifi_setup_intro.wav"
    assert events[1] == "prepare"
    assert events[-1] == "finish"
    assert "sound:wifi_connect_success.wav" in events


def test_finish_runs_on_timeout_too():
    """The robot must return to sleep even when no QR was ever shown."""
    prov, seen = make_provisioner(payloads=None, scan_timeout_s=0.1)
    prov.start()
    wait_done(prov)
    assert prov.status().state == ProvisioningState.FAILED
    assert seen["events"][-1] == "finish"


def test_finish_runs_when_camera_is_unavailable():
    """The robot must return to sleep even if the camera never opened."""

    class BrokenCameraFactory:
        def __call__(self):
            raise RuntimeError("no camera")

    prov, seen = make_provisioner()
    prov._camera_factory = BrokenCameraFactory()
    prov.start()
    wait_done(prov)
    assert prov.status().state == ProvisioningState.UNAVAILABLE
    assert seen["events"][-1] == "finish"


def test_scan_clock_starts_after_the_intro():
    """The one-minute countdown must not tick while the narration plays."""
    prov, seen = make_provisioner(
        payloads=["WIFI:T:WPA;S:net;P:pw;;"], intro_wait_s=0.2, scan_timeout_s=0.3
    )
    t0 = time.monotonic()
    prov.start()
    seen["started_t"] = t0
    wait_done(prov)
    assert prov.status().state == ProvisioningState.SUCCESS
    # no decode may happen before the intro is over
    assert seen["first_decode_t"] - t0 >= 0.2


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
    assert "wifi_failed_no_qr.wav" in seen["sounds"]


def test_each_failure_mode_has_its_own_voice_line():
    """Bad password (hotspot revert) and connect timeout speak differently."""
    prov, seen = make_provisioner(
        payloads=["WIFI:T:WPA;S:net;P:badpw;;"],
        status_after_connect=[("busy", None), ("hotspot", None)],
    )
    prov.start()
    wait_done(prov)
    assert "wifi_failed_bad_credentials.wav" in seen["sounds"]

    prov, seen = make_provisioner(
        payloads=["WIFI:T:WPA;S:net;P:pw;;"],
        status_after_connect=[("busy", None)],  # never comes up
        connect_timeout_s=0.1,
    )
    prov.start()
    wait_done(prov)
    assert "wifi_failed_connect_timeout.wav" in seen["sounds"]


def test_fails_when_connection_never_comes_up():
    prov, seen = make_provisioner(
        payloads=["WIFI:T:WPA;S:net;P:pw;;"],
        status_after_connect=[("busy", None)],  # stuck busy forever
        connect_timeout_s=0.1,
    )
    prov.start()
    wait_done(prov)
    assert prov.status().state == ProvisioningState.FAILED


def test_fails_when_daemon_reverts_to_hotspot():
    # Same semantics as the desktop app onboarding: after busy, landing on
    # hotspot means the daemon gave up and reverted (e.g. wrong password).
    prov, seen = make_provisioner(
        payloads=["WIFI:T:WPA;S:net;P:badpw;;"],
        status_after_connect=[("busy", None), ("busy", None), ("hotspot", None)],
    )
    prov.start()
    wait_done(prov)
    assert prov.status().state == ProvisioningState.FAILED
    assert "hotspot" in prov.status().detail


def test_success_requires_the_right_network():
    # Connected to some OTHER wlan does not count as success for this ssid.
    prov, seen = make_provisioner(
        payloads=["WIFI:T:WPA;S:net;P:pw;;"],
        status_after_connect=[("busy", None), ("wlan", "other-net")],
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
