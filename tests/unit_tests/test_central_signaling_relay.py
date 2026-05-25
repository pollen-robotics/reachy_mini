"""Unit tests for :mod:`reachy_mini.media.central_signaling_relay`.

These tests cover only the pure helpers exposed on the relay module:
the welcome-message negotiation logic that picks the heartbeat
cadence. Anything that would require driving the asyncio loop, the
local GStreamer websocket, or the central HTTP server is out of
scope here (and is currently tracked as integration-test tech debt
in the same way as the robot_app_lock test suite).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import pytest

from reachy_mini.media.central_signaling_relay import (
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_MAX_INTERVAL,
    HEARTBEAT_MIN_INTERVAL,
    CentralSignalingRelay,
    _clamp_heartbeat_interval,
)

# ---------------------------------------------------------------------------
# _clamp_heartbeat_interval
# ---------------------------------------------------------------------------


def test_clamp_passes_through_in_range_value() -> None:
    """A value already inside [MIN, MAX] is returned unchanged."""
    assert _clamp_heartbeat_interval(5.0) == 5.0


def test_clamp_floors_below_min() -> None:
    """A sub-floor value is raised to ``HEARTBEAT_MIN_INTERVAL``."""
    assert _clamp_heartbeat_interval(0.1) == HEARTBEAT_MIN_INTERVAL


def test_clamp_ceilings_above_max() -> None:
    """An over-ceiling value is lowered to ``HEARTBEAT_MAX_INTERVAL``."""
    assert _clamp_heartbeat_interval(600.0) == HEARTBEAT_MAX_INTERVAL


# ---------------------------------------------------------------------------
# Heartbeat interval negotiation
# ---------------------------------------------------------------------------
#
# The cascade is documented in `_negotiate_heartbeat_interval`'s
# docstring. We assert each rung explicitly so a future refactor that
# inverts priorities (or drops the clamp) trips a regression here
# rather than at runtime against a real central deployment.


def test_negotiate_uses_recommended_when_present() -> None:
    """Rung 1: the canonical field wins over everything else."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {
            "type": "welcome",
            "peerId": "abc",
            "recommended_heartbeat_interval_seconds": 4.0,
            "lease_seconds": 30.0,  # would otherwise yield 10.0
        }
    )
    assert result == 4.0


def test_negotiate_falls_back_to_lease_over_three() -> None:
    """Rung 2: an older central exposing only `lease_seconds` is honored."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {"type": "welcome", "peerId": "abc", "lease_seconds": 15.0}
    )
    assert result == pytest.approx(5.0)


def test_negotiate_falls_back_to_default_when_no_negotiation() -> None:
    """Rung 3: a pre-negotiation central gets the conservative default."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {"type": "welcome", "peerId": "abc"}
    )
    assert result == HEARTBEAT_DEFAULT_INTERVAL


def test_negotiate_clamps_below_floor() -> None:
    """Rung 1 is subject to the safety clamp on its lower bound."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {"recommended_heartbeat_interval_seconds": 0.1}
    )
    assert result == HEARTBEAT_MIN_INTERVAL


def test_negotiate_clamps_above_ceiling() -> None:
    """Rung 1 is subject to the safety clamp on its upper bound."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {"recommended_heartbeat_interval_seconds": 600.0}
    )
    assert result == HEARTBEAT_MAX_INTERVAL


def test_negotiate_clamps_lease_derived_value() -> None:
    """Rung 2 is also subject to the safety clamp."""
    # lease=600s would derive 200s, well above the 60s ceiling.
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {"lease_seconds": 600.0}
    )
    assert result == HEARTBEAT_MAX_INTERVAL


def test_negotiate_ignores_non_numeric_recommended() -> None:
    """A garbled `recommended_*` field falls through to the next rung."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {
            "recommended_heartbeat_interval_seconds": "soon",
            "lease_seconds": 15.0,
        }
    )
    # Must come from `lease_seconds / 3`, not from the bad string.
    assert result == pytest.approx(5.0)


def test_negotiate_ignores_non_positive_recommended() -> None:
    """A zero / negative `recommended_*` value falls through to the next rung."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {
            "recommended_heartbeat_interval_seconds": -1,
            "lease_seconds": 15.0,
        }
    )
    assert result == pytest.approx(5.0)


def test_negotiate_ignores_non_positive_lease() -> None:
    """A zero / negative `lease_seconds` falls through to the default."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval({"lease_seconds": 0})
    assert result == HEARTBEAT_DEFAULT_INTERVAL


def test_negotiate_ignores_non_numeric_lease() -> None:
    """A garbled `lease_seconds` value falls through to the default."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {"lease_seconds": "fifteen"}
    )
    assert result == HEARTBEAT_DEFAULT_INTERVAL


def test_negotiate_accepts_int_recommended() -> None:
    """JSON ``recommended_*`` may arrive as int (no decimal); we still accept it."""
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {"recommended_heartbeat_interval_seconds": 5}
    )
    assert result == 5.0
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _build_producer_meta
# ---------------------------------------------------------------------------
#
# These tests are the contract guard for the meta dict the daemon
# advertises to central listeners. They lock down the exact wire shape
# the mobile app reads. A future change extending the meta (install_id,
# capabilities, ...) MUST add new tests here, not edit the existing
# ones - listening clients at older versions rely on these fields not
# disappearing.


def _make_relay(
    robot_name: str = "reachymini", transport: str = "wifi"
) -> CentralSignalingRelay:
    """Construct a relay without starting the asyncio plumbing.

    The constructor only stores fields; ``start()`` is what binds to the
    websocket and HTTP client. Calling ``__init__`` directly is therefore
    safe for pure-function tests.
    """
    return CentralSignalingRelay(robot_name=robot_name, transport=transport)


def test_meta_carries_robot_name() -> None:
    """The relay carries ``robot_name`` into ``meta.name`` verbatim."""
    relay = _make_relay(robot_name="Sparky")
    assert relay._build_producer_meta()["name"] == "Sparky"


def test_meta_carries_transport_wifi_by_default() -> None:
    """The relay defaults to ``transport='wifi'`` when no override is passed.

    This matches the Pi-side autonomous daemon, which is the common
    case (the desktop tray is the one that has to opt in).
    """
    relay = _make_relay()
    assert relay._build_producer_meta()["transport"] == "wifi"


def test_meta_carries_transport_usb_when_set() -> None:
    """The relay forwards ``transport='usb'`` verbatim into ``meta``."""
    relay = _make_relay(transport="usb")
    assert relay._build_producer_meta()["transport"] == "usb"


def test_meta_forwards_unknown_transport_value_verbatim() -> None:
    """``transport`` is intentionally a free-form string.

    Future fronts (``"ethernet"``, ``"sim"``, ``"mockup"``, ...) must
    propagate to listeners without a relay change. Listeners that
    don't recognise the value fall back to "Wi-Fi" by convention.
    """
    relay = _make_relay(transport="ethernet")
    assert relay._build_producer_meta()["transport"] == "ethernet"


def test_meta_used_by_producer_status_payload() -> None:
    """``_producer_status_payload`` MUST embed the meta verbatim.

    The payload is the canonical envelope, ``_build_producer_meta``
    is the source of truth for the dict's content. Any divergence
    would cause the heartbeat re-emission to drift from the initial
    registration.
    """
    relay = _make_relay(robot_name="r1", transport="usb")
    payload = relay._producer_status_payload()
    assert payload["type"] == "setPeerStatus"
    assert payload["roles"] == ["producer"]
    assert payload["meta"] == relay._build_producer_meta()


# ---------------------------------------------------------------------------
# meta.hardware_id (stable per-robot identity)
# ---------------------------------------------------------------------------
#
# ``hardware_id`` is sourced via ``get_hardware_id()`` which reads the
# Pollen audio device's USB serial from sysfs. We monkeypatch that
# function so the tests exercise the relay's ``meta`` plumbing, not
# the underlying serial-reading helper (which has its own coverage).


def test_meta_includes_hardware_id_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``get_hardware_id()`` returns a string, the meta carries it.

    Listeners use this as the stable cross-source identity (BLE GATT
    / central / loopback HTTP all expose the same value).
    """
    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "get_hardware_id", lambda: "abc123def456")
    relay = _make_relay()
    meta = relay._build_producer_meta()
    assert meta["hardware_id"] == "abc123def456"


def test_meta_omits_hardware_id_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no Reachy is attached, the ``hardware_id`` field is absent.

    ``get_hardware_id() is None`` MUST result in the field being
    omitted from the meta - never serialised as ``null`` or as the
    literal ``"unknown"``. Consumers branch on
    ``"hardware_id" in meta`` to decide whether to use the stable id
    or fall back to ``peerId`` / BLE address.
    """
    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "get_hardware_id", lambda: None)
    relay = _make_relay()
    meta = relay._build_producer_meta()
    assert "hardware_id" not in meta


def test_meta_shape_with_hardware_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lock the full set of keys when ``hardware_id`` is available.

    A listener written today must not be broken by an accidental
    field rename. Add new keys, never remove or rename.
    """
    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "get_hardware_id", lambda: "deadbeef00112233")
    relay = _make_relay()
    meta = relay._build_producer_meta()
    assert set(meta.keys()) == {"name", "transport", "hardware_id"}


def test_meta_shape_without_hardware_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lock the minimal key set when no Reachy is attached.

    Same lock intent as ``test_meta_shape_with_hardware_id``: a
    future PR adding a new field MUST extend both shape tests rather
    than relax them.
    """
    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "get_hardware_id", lambda: None)
    relay = _make_relay()
    meta = relay._build_producer_meta()
    assert set(meta.keys()) == {"name", "transport"}


# ---------------------------------------------------------------------------
# notify_peer_session_failed (daemon-side WebRTC watchdog -> central)
# ---------------------------------------------------------------------------
#
# These tests exercise the relay-side leg of the negotiation
# watchdog (the GStreamer-side leg lives in
# `test_media_server_watchdog.py`). The watchdog fires when libnice
# (or any other piece of the WebRTC plumbing) leaves the peer
# stuck mid-negotiation; the relay translates that into an
# `endSession` message routed at the central session id, so the
# JS client gets a typed rejection.
#
# We drive the inner coroutine `_notify_peer_session_failed`
# directly with `asyncio.run` and replace `_send_to_central` with a
# stub journal so we can assert on the wire-level shape without
# spinning up an HTTP session.


class _SendJournal:
    """Capture ``_send_to_central`` calls for inspection in tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def __call__(self, msg: dict[str, Any]) -> None:
        self.sent.append(msg)


def _make_relay_with_session(
    *,
    local_session_id: str = "local-1",
    central_session_id: str = "central-1",
    client_peer_id: str = "client-peer-1",
) -> tuple[CentralSignalingRelay, _SendJournal]:
    """Build a relay with one in-flight session and a stubbed central send.

    Mirrors the state shape produced by
    ``_process_local_message("sessionStarted")`` plus
    ``_process_central_message("startSession")``.
    """
    relay = _make_relay()
    journal = _SendJournal()
    relay._send_to_central = journal  # type: ignore[method-assign]
    relay._local_to_central_session[local_session_id] = central_session_id
    relay._central_to_local_session[central_session_id] = local_session_id
    relay._session_to_local_peer[central_session_id] = client_peer_id
    return relay, journal


def test_notify_peer_session_failed_sends_endsession_to_central() -> None:
    """The wire shape of the failure message uses the central session id."""
    relay, journal = _make_relay_with_session()

    asyncio.run(
        relay._notify_peer_session_failed(
            "local-1",
            "ice_negotiation_timeout",
            {"ice_state": "checking", "elapsed_s": 12.0},
        )
    )

    assert len(journal.sent) == 1
    msg = journal.sent[0]
    assert msg["type"] == "endSession"
    assert msg["sessionId"] == "central-1"
    assert msg["reason"] == "ice_negotiation_timeout"


def test_notify_peer_session_failed_clears_session_bookkeeping() -> None:
    """Session bookkeeping is cleared after a failure is reported to central."""
    relay, _journal = _make_relay_with_session(
        local_session_id="local-x",
        central_session_id="central-x",
        client_peer_id="peer-x",
    )

    asyncio.run(relay._notify_peer_session_failed("local-x", "any-reason", {}))

    assert "local-x" not in relay._local_to_central_session
    assert "central-x" not in relay._central_to_local_session
    assert "central-x" not in relay._session_to_local_peer


def test_notify_peer_session_failed_drops_pending_central_session() -> None:
    """A pending central session for the failed peer is dropped from the queue."""
    relay, journal = _make_relay_with_session(
        local_session_id="local-p",
        central_session_id="central-p",
    )
    relay._pending_central_sessions.append("central-p")

    asyncio.run(relay._notify_peer_session_failed("local-p", "any-reason", {}))

    assert "central-p" not in relay._pending_central_sessions
    assert journal.sent[0]["sessionId"] == "central-p"


def test_notify_peer_session_failed_no_op_when_session_unknown() -> None:
    """A watchdog firing for an untracked session is a clean no-op."""
    relay = _make_relay()
    journal = _SendJournal()
    relay._send_to_central = journal  # type: ignore[method-assign]

    asyncio.run(relay._notify_peer_session_failed("ghost", "any", {}))

    assert journal.sent == []


def test_notify_peer_session_failed_logs_diagnostic_in_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The diagnostic dict is serialised to the daemon log for grepping."""
    relay, _journal = _make_relay_with_session()
    diagnostic = {"ice_state": "checking", "conn_state": "new", "elapsed_s": 12.0}

    with caplog.at_level(logging.ERROR):
        asyncio.run(
            relay._notify_peer_session_failed(
                "local-1", "ice_negotiation_timeout", diagnostic
            )
        )

    matching = [
        rec
        for rec in caplog.records
        if "daemon-side WebRTC failure" in rec.getMessage()
    ]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "checking" in msg
    assert "ice_negotiation_timeout" in msg


def test_public_notify_no_op_without_running_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The public entry point drops the call when the relay loop is gone."""
    relay = _make_relay()
    assert relay._thread_loop is None

    with caplog.at_level(logging.DEBUG):
        relay.notify_peer_session_failed("local-1", "any", {})

    debug_lines = [
        rec for rec in caplog.records if "relay loop not running" in rec.getMessage()
    ]
    assert len(debug_lines) == 1


def test_public_notify_schedules_work_on_relay_loop() -> None:
    """The public entry point hops failure handling onto the relay's loop."""
    relay, journal = _make_relay_with_session()

    loop = asyncio.new_event_loop()
    stop = threading.Event()

    async def _block_until(evt: threading.Event) -> None:
        while not evt.is_set():
            await asyncio.sleep(0.01)

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_block_until(stop))
        finally:
            loop.close()

    relay._thread_loop = loop
    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    try:
        relay.notify_peer_session_failed(
            "local-1", "ice_negotiation_timeout", {"ice_state": "checking"}
        )
        for _ in range(50):
            if journal.sent:
                break
            time.sleep(0.02)
    finally:
        stop.set()
        t.join(timeout=2.0)

    assert len(journal.sent) == 1
    assert journal.sent[0]["sessionId"] == "central-1"
    assert journal.sent[0]["reason"] == "ice_negotiation_timeout"
