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
import json
import logging
import threading
import time
from typing import Any, Optional

import pytest

from reachy_mini.daemon.robot_app_lock import RobotAppLock, RobotAppLockState
from reachy_mini.media.central_signaling_relay import (
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_MAX_INTERVAL,
    HEARTBEAT_MIN_INTERVAL,
    MAX_RECONNECT_INTERVAL,
    RECONNECT_BACKOFF_JITTER,
    RECONNECT_INTERVAL,
    CentralSignalingRelay,
    RelayState,
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


def test_relay_rejects_plaintext_remote_central() -> None:
    """The relay must not send its bearer token over remote HTTP."""
    with pytest.raises(ValueError):
        CentralSignalingRelay(central_uri="http://central.example")


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
    assert set(meta.keys()) == {
        "name",
        "transport",
        "hardware_id",
        "session_liveness",
        "session_active",
    }


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
    assert set(meta.keys()) == {
        "name",
        "transport",
        "session_liveness",
        "session_active",
    }


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
    relay._send_to_local = _SendJournal()  # type: ignore[method-assign]
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


def test_notify_peer_session_failed_resolves_relay_consumer_id_to_newest_session() -> (
    None
):
    """Relayed sessions all share the relay's own consumer id.

    webrtcsink reports the consumer's signalling peer id, which is the
    relay's local peer id for every relayed session - so the failure is
    pinned on the newest session, the one still negotiating.
    """
    relay, journal = _make_relay_with_session(
        local_session_id="local-old", central_session_id="central-old"
    )
    relay._local_peer_id = "relay-consumer"
    relay._local_to_central_session["local-new"] = "central-new"
    relay._central_to_local_session["central-new"] = "local-new"

    asyncio.run(
        relay._notify_peer_session_failed(
            "relay-consumer", "ice_negotiation_timeout", {}
        )
    )

    assert journal.sent == [
        {
            "type": "endSession",
            "sessionId": "central-new",
            "reason": "ice_negotiation_timeout",
        }
    ]
    assert "local-new" not in relay._local_to_central_session
    assert relay._local_to_central_session == {"local-old": "central-old"}


def test_notify_peer_session_failed_ends_the_session_locally() -> None:
    """A session stuck before ICE is ended on the GStreamer side too."""
    relay, _journal = _make_relay_with_session()
    local = relay._send_to_local

    asyncio.run(relay._notify_peer_session_failed("local-1", "any", {}))

    assert local.sent == [{"type": "endSession", "sessionId": "local-1"}]  # type: ignore[attr-defined]


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


def test_run_loop_backs_off_when_connect_returns_in_error_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 401-style failure (ERROR state, no exception) still gets a backoff."""
    monkeypatch.setattr(
        "reachy_mini.media.central_signaling_relay.RECONNECT_INTERVAL", 0.05
    )

    relay = _make_relay()
    relay._running = True

    call_count = 0

    async def fake_connect_and_relay() -> None:
        nonlocal call_count
        call_count += 1
        relay._set_state(
            RelayState.ERROR, "Authentication failed - token may be invalid"
        )
        if call_count >= 3:
            # Stop before the post-call backoff bookkeeping for this
            # iteration runs, mirroring `stop()` racing the next attempt.
            relay._running = False

    relay._connect_and_relay = fake_connect_and_relay  # type: ignore[method-assign]

    start = time.monotonic()
    asyncio.run(relay._run_loop())
    elapsed = time.monotonic() - start

    assert call_count == 3
    assert relay._connection_attempts == 2
    assert elapsed >= 0.1


def test_run_loop_backs_off_when_connect_ends_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attempt that dies right away (no ERROR state) still gets a backoff."""
    monkeypatch.setattr(
        "reachy_mini.media.central_signaling_relay.RECONNECT_INTERVAL", 0.05
    )

    relay = _make_relay()
    relay._running = True

    call_count = 0

    async def fake_connect_and_relay() -> None:
        nonlocal call_count
        call_count += 1
        # e.g. central accepts the SSE stream and closes it at once.
        relay._set_state(RelayState.RECONNECTING, "Connection lost, reconnecting...")
        if call_count >= 3:
            relay._running = False

    relay._connect_and_relay = fake_connect_and_relay  # type: ignore[method-assign]

    start = time.monotonic()
    asyncio.run(relay._run_loop())
    elapsed = time.monotonic() - start

    assert call_count == 3
    assert relay._connection_attempts == 2
    assert elapsed >= 0.1


def test_connect_after_token_update_does_not_tear_itself_down() -> None:
    """A login's token signal is consumed, not replayed on every reconnect.

    Regression: the event stayed set after the login reconnect, so each new
    attempt's token watcher fired at once and closed its own HTTP session.
    """
    relay = _make_relay()
    relay._running = True
    relay.hf_token = "hf_token"
    relay._refresh_token = lambda force_refresh=False: relay.hf_token  # type: ignore[method-assign]
    relay._token_updated.set()

    async def forever() -> None:
        await asyncio.Event().wait()

    async def fake_connect_local() -> None:
        relay._local_ws = _FakeWebsocket()  # type: ignore[assignment]
        relay._local_task = asyncio.create_task(forever())

    relay._connect_local = fake_connect_local  # type: ignore[method-assign]
    relay._handle_central_sse = forever  # type: ignore[method-assign]
    relay._producer_health_loop = forever  # type: ignore[method-assign]
    relay._heartbeat_loop = forever  # type: ignore[method-assign]

    async def run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(relay._connect_and_relay(), timeout=0.2)

    asyncio.run(run())


def test_reconnect_delay_grows_and_is_capped() -> None:
    """``_reconnect_delay`` backs off exponentially and stays under the cap."""
    relay = _make_relay()

    relay._connection_attempts = 1
    d1 = relay._reconnect_delay()
    relay._connection_attempts = 2
    d2 = relay._reconnect_delay()
    relay._connection_attempts = 3
    d3 = relay._reconnect_delay()

    # Strictly increasing: the base doubles each step, and the additive
    # jitter (<=10%) cannot make a later attempt shorter than an earlier one.
    assert RECONNECT_INTERVAL <= d1 < d2 < d3

    # Capped on long outages, even with the maximum jitter applied.
    relay._connection_attempts = 1000
    assert (
        relay._reconnect_delay()
        <= MAX_RECONNECT_INTERVAL * (1 + RECONNECT_BACKOFF_JITTER) + 1e-9
    )


def test_reconnect_delay_stays_short_while_sessions_are_kept() -> None:
    """Clients of kept sessions need the robot back on central quickly."""
    from reachy_mini.media.central_signaling_relay import (
        KEPT_SESSIONS_MAX_RECONNECT_INTERVAL,
    )

    relay = _make_relay()
    relay._connection_attempts = 1000
    relay._local_to_central_session["local-1"] = "s1"
    assert (
        relay._reconnect_delay()
        <= KEPT_SESSIONS_MAX_RECONNECT_INTERVAL * (1 + RECONNECT_BACKOFF_JITTER) + 1e-9
    )


# ---------------------------------------------------------------------------
# startSession robot-lock gating (free / local_app control-only / busy)
# ---------------------------------------------------------------------------


def _relay_with_lock(
    lock: RobotAppLock,
) -> tuple[CentralSignalingRelay, _SendJournal, _SendJournal]:
    """Relay wired to a real lock, with central + local sends journaled."""
    relay = _make_relay()
    relay._robot_app_lock = lock
    central, local = _SendJournal(), _SendJournal()
    relay._send_to_central = central  # type: ignore[method-assign]
    relay._send_to_local = local  # type: ignore[method-assign]
    return relay, central, local


def _start_session(relay: CentralSignalingRelay) -> None:
    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "p", "sessionId": "s1"}
        )
    )


def test_startsession_acquires_lock_when_free() -> None:
    """A remote peer takes remote_session when the robot is free."""
    lock = RobotAppLock()
    relay, central, local = _relay_with_lock(lock)
    _start_session(relay)
    assert lock.status().state == RobotAppLockState.REMOTE_SESSION
    assert not any(m.get("type") == "endSession" for m in central.sent)
    assert any(m.get("type") == "list" for m in local.sent)  # session proceeds


def _local_starts(local: _SendJournal) -> int:
    return sum(1 for m in local.sent if m.get("type") == "startSession")


def test_quick_redial_starts_one_local_session() -> None:
    """Two list replies for one pending request start ONE local session.

    Right after a daemon restart a phone can send two requests about 1 ms
    apart: central ends the first at once, the second stays pending, and
    both requests' list replies arrive afterwards. Each used to start a
    local session for the pending request: an orphan webrtcbin whose data
    channel then replaced the real one.
    """
    relay, _central, local = _relay_with_lock(RobotAppLock())
    producers = {"type": "list", "producers": [{"id": "producer-1"}]}

    async def run() -> None:
        await relay._process_central_message(
            {"type": "startSession", "peerId": "p", "sessionId": "s1"}
        )
        await relay._process_central_message({"type": "endSession", "sessionId": "s1"})
        await relay._process_central_message(
            {"type": "startSession", "peerId": "p", "sessionId": "s2"}
        )
        await relay._process_local_message(producers)
        await relay._process_local_message(producers)

    asyncio.run(run())
    assert _local_starts(local) == 1
    asyncio.run(
        relay._process_local_message({"type": "sessionStarted", "sessionId": "local-2"})
    )
    assert relay._central_to_local_session == {"s2": "local-2"}


def test_orphan_local_session_is_ended() -> None:
    """A local session nobody waits for is ended, whatever ordering made it.

    Here the first request's list reply lands before central ends that
    request, so a start goes out for each request; the second local
    sessionStarted then finds nothing pending.
    """
    relay, _central, local = _relay_with_lock(RobotAppLock())
    producers = {"type": "list", "producers": [{"id": "producer-1"}]}

    async def run() -> None:
        await relay._process_central_message(
            {"type": "startSession", "peerId": "p", "sessionId": "s1"}
        )
        await relay._process_local_message(producers)
        await relay._process_central_message({"type": "endSession", "sessionId": "s1"})
        await relay._process_central_message(
            {"type": "startSession", "peerId": "p", "sessionId": "s2"}
        )
        await relay._process_local_message(producers)
        await relay._process_local_message(
            {"type": "sessionStarted", "sessionId": "L1"}
        )
        await relay._process_local_message(
            {"type": "sessionStarted", "sessionId": "L2"}
        )

    asyncio.run(run())
    # The second list reply found a start still unanswered: no second start.
    assert _local_starts(local) == 1
    assert relay._central_to_local_session == {"s2": "L1"}
    # A stray local session (injected here as L2) is ended all the same.
    assert {"type": "endSession", "sessionId": "L2"} in local.sent


def test_request_waiting_for_a_late_producer_is_started() -> None:
    """A request accepted before the media pipeline registered still starts.

    After a daemon restart the relay can be back on central before
    GStreamer's producer is up: the list reply is empty, and the request
    used to stay pending forever, the robot answering "busy" to everyone.
    """
    relay, _central, local = _relay_with_lock(RobotAppLock())

    async def run() -> None:
        await relay._process_central_message(
            {"type": "startSession", "peerId": "p", "sessionId": "s1"}
        )
        await relay._process_local_message({"type": "list", "producers": []})
        assert _local_starts(local) == 0
        await relay._process_local_message(
            {"type": "peerStatusChanged", "peerId": "producer-1", "roles": ["producer"]}
        )

    asyncio.run(run())
    assert _local_starts(local) == 1


def test_lost_local_start_is_resent_after_the_answer_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A start the local server never answered no longer blocks the next one."""
    import reachy_mini.media.central_signaling_relay as relay_module

    relay, _central, local = _relay_with_lock(RobotAppLock())
    producers = {"type": "list", "producers": [{"id": "producer-1"}]}
    clock = [1000.0]
    monkeypatch.setattr(relay_module.time, "monotonic", lambda: clock[0])
    _start_session(relay)
    asyncio.run(relay._process_local_message(producers))
    asyncio.run(relay._process_local_message(producers))
    assert _local_starts(local) == 1
    clock[0] += relay_module.LOCAL_START_ANSWER_S + 1
    asyncio.run(relay._process_local_message(producers))
    assert _local_starts(local) == 2


def test_eviction_only_ends_the_sessions_up_when_it_was_asked() -> None:
    """A session accepted after the local app took the robot is left alone."""
    relay, central, local = _make_relay_with_journals()
    relay._central_to_local_session.update({"old": "L-old", "new": "L-new"})
    relay._local_to_central_session.update({"L-old": "old", "L-new": "new"})

    asyncio.run(
        relay._tear_down_active_sessions(reason="local_app_started", only={"old"})
    )

    assert central.sent == [
        {"type": "endSession", "sessionId": "old", "reason": "local_app_started"}
    ]
    assert local.sent == [{"type": "endSession", "sessionId": "L-old"}]
    assert relay._central_to_local_session == {"new": "L-new"}
    assert relay._local_to_central_session == {"L-new": "new"}


def test_a_new_producer_is_not_blocked_by_starts_sent_to_the_old_one() -> None:
    """Starts sent to a producer that left cover nothing once a new one registers.

    The local server answers such a start with an error the relay ignores;
    counted as unanswered, it used to keep the pending request from being
    started on the re-registered producer for 5 s (and nothing retried).
    """
    relay, _central, local = _relay_with_lock(RobotAppLock())

    async def run() -> None:
        await relay._process_central_message(
            {"type": "startSession", "peerId": "p", "sessionId": "s1"}
        )
        await relay._process_local_message(
            {"type": "list", "producers": [{"id": "old-producer"}]}
        )
        assert _local_starts(local) == 1
        await relay._process_local_message(
            {
                "type": "peerStatusChanged",
                "peerId": "new-producer",
                "roles": ["producer"],
            }
        )

    asyncio.run(run())
    assert _local_starts(local) == 2
    assert local.sent[-1] == {"type": "startSession", "peerId": "new-producer"}


def test_startsession_control_only_while_local_app_runs() -> None:
    """While a local app holds the robot, a remote peer connects control-only."""
    lock = RobotAppLock()
    lock.acquire_local_keeping_remote("reachy_mini_conversation_app")
    relay, central, local = _relay_with_lock(lock)
    _start_session(relay)
    # accepted (session proceeds) WITHOUT taking the lock or ending the session
    assert lock.status().state == RobotAppLockState.LOCAL_APP
    assert not any(m.get("type") == "endSession" for m in central.sent)
    assert any(m.get("type") == "list" for m in local.sent)


def test_startsession_refused_when_another_remote_holds() -> None:
    """A second remote peer is refused while another remote session owns the robot."""
    lock = RobotAppLock()
    assert lock.try_acquire_remote("other") is True
    relay, central, local = _relay_with_lock(lock)
    _start_session(relay)
    assert any(
        m.get("type") == "endSession" and m.get("reason") == "robot_busy"
        for m in central.sent
    )


# ---------------------------------------------------------------------------
# Message dispatch: _process_central_message / _process_local_message
# ---------------------------------------------------------------------------
#
# The relay's job is to shuttle WebRTC signalling between the central server
# (SSE/HTTP) and the local GStreamer webrtcsink (websocket), translating
# session ids across the two namespaces and enforcing single-session +
# robot-lock gating. These tests drive both dispatchers directly with
# `asyncio.run`, stubbing the two send methods with journals so we assert the
# relayed wire shape with no sockets, no HTTP, and no GStreamer.


class _FakeLock:
    """Minimal RobotAppLock stand-in recording acquire/release calls."""

    def __init__(self, *, acquire: bool = True) -> None:
        self._acquire = acquire
        self.acquired = 0
        self.released = 0
        self.handoff_releases = 0

    def try_acquire_remote(self, app_name: str) -> bool:
        self.acquired += 1
        return self._acquire

    def release_remote(self, *, expect_handoff: bool = False) -> None:
        self.released += 1
        if expect_handoff:
            self.handoff_releases += 1


def _make_relay_with_journals(
    *, robot_app_lock: Any = None
) -> tuple[CentralSignalingRelay, _SendJournal, _SendJournal]:
    """Relay with both send legs stubbed and a truthy local websocket.

    The central `peer` handler gates on ``self._local_ws`` being set, so we
    give it a sentinel; the actual send is captured by the local journal.
    """
    relay = CentralSignalingRelay(robot_app_lock=robot_app_lock)
    central, local = _SendJournal(), _SendJournal()
    relay._send_to_central = central  # type: ignore[method-assign]
    relay._send_to_local = local  # type: ignore[method-assign]
    relay._local_ws = object()  # type: ignore[assignment]
    return relay, central, local


# ---- central -> local direction ----


def test_central_welcome_registers_producer_and_connects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`welcome` stores the peer id, registers as producer, flips to CONNECTED."""
    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "get_hardware_id", lambda: None)
    relay, central, _local = _make_relay_with_journals()

    asyncio.run(relay._process_central_message({"type": "welcome", "peerId": "C1"}))

    assert relay._central_peer_id == "C1"
    assert relay.state == RelayState.CONNECTED
    assert len(central.sent) == 1
    assert central.sent[0]["type"] == "setPeerStatus"
    assert central.sent[0]["roles"] == ["producer"]


def test_central_start_session_requests_local_list() -> None:
    """A fresh `startSession` is tracked as pending and asks local for its list."""
    relay, central, local = _make_relay_with_journals()

    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "consumer", "sessionId": "cs1"}
        )
    )

    assert relay._session_to_local_peer == {"cs1": "consumer"}
    assert relay._pending_central_sessions == ["cs1"]
    assert local.sent == [{"type": "list"}]
    assert central.sent == []


def test_central_start_session_rejected_when_busy() -> None:
    """A second session is rejected with robot_busy_local while one is active."""
    relay, central, local = _make_relay_with_journals()
    relay._central_to_local_session["existing"] = "local-existing"

    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "consumer", "sessionId": "cs2"}
        )
    )

    assert local.sent == []
    assert central.sent == [
        {"type": "endSession", "sessionId": "cs2", "reason": "robot_busy_local"}
    ]
    assert "cs2" not in relay._session_to_local_peer


def test_central_peer_translates_session_id_to_local() -> None:
    """A `peer` message is relayed to local with the local session id + sdp."""
    relay, _central, local = _make_relay_with_journals()
    relay._central_to_local_session["cs1"] = "ls1"

    asyncio.run(
        relay._process_central_message(
            {"type": "peer", "sessionId": "cs1", "sdp": {"type": "offer"}}
        )
    )

    assert local.sent == [
        {"type": "peer", "sessionId": "ls1", "sdp": {"type": "offer"}}
    ]


def test_central_peer_dropped_when_no_mapping() -> None:
    """A `peer` for an unknown session is dropped, not forwarded."""
    relay, _central, local = _make_relay_with_journals()

    asyncio.run(
        relay._process_central_message(
            {"type": "peer", "sessionId": "unknown", "ice": {"candidate": "x"}}
        )
    )

    assert local.sent == []


def test_central_end_session_cleans_up_and_releases_lock() -> None:
    """`endSession` clears both maps, forwards to local, releases the lock."""
    lock = _FakeLock()
    relay, _central, local = _make_relay_with_journals(robot_app_lock=lock)
    relay._session_to_local_peer["cs1"] = "consumer"
    relay._central_to_local_session["cs1"] = "ls1"
    relay._local_to_central_session["ls1"] = "cs1"

    asyncio.run(
        relay._process_central_message({"type": "endSession", "sessionId": "cs1"})
    )

    assert relay._central_to_local_session == {}
    assert relay._local_to_central_session == {}
    assert relay._session_to_local_peer == {}
    assert local.sent == [{"type": "endSession", "sessionId": "ls1"}]
    assert lock.released == 1
    # A deliberate hang-up: flagged as a hand-off so the daemon gives a
    # successor session time to arrive before sleeping the robot.
    assert lock.handoff_releases == 1


# ---- local -> central direction ----


def test_local_welcome_registers_as_listener() -> None:
    """Local `welcome` stores the peer id and registers as a listener."""
    relay, _central, local = _make_relay_with_journals()

    asyncio.run(relay._process_local_message({"type": "welcome", "peerId": "L1"}))

    assert relay._local_peer_id == "L1"
    assert len(local.sent) == 1
    assert local.sent[0]["type"] == "setPeerStatus"
    assert local.sent[0]["roles"] == ["listener"]


def test_local_list_starts_pending_sessions() -> None:
    """A producer `list` starts a local session for each pending central one."""
    relay, _central, local = _make_relay_with_journals()
    relay._pending_central_sessions.append("cs1")

    asyncio.run(
        relay._process_local_message({"type": "list", "producers": [{"id": "P1"}]})
    )

    assert relay._local_producer_id == "P1"
    assert local.sent == [{"type": "startSession", "peerId": "P1"}]


def test_local_session_started_maps_ids() -> None:
    """`sessionStarted` binds the local session id to the pending central one."""
    relay, _central, _local = _make_relay_with_journals()
    relay._pending_central_sessions.append("cs1")

    asyncio.run(
        relay._process_local_message({"type": "sessionStarted", "sessionId": "ls1"})
    )

    assert relay._local_to_central_session == {"ls1": "cs1"}
    assert relay._central_to_local_session == {"cs1": "ls1"}
    assert relay._pending_central_sessions == []


def test_local_peer_translates_session_id_to_central() -> None:
    """A local `peer` is relayed to central with the central session id."""
    relay, central, _local = _make_relay_with_journals()
    relay._local_to_central_session["ls1"] = "cs1"

    asyncio.run(
        relay._process_local_message(
            {"type": "peer", "sessionId": "ls1", "ice": {"candidate": "c"}}
        )
    )

    assert central.sent == [
        {"type": "peer", "sessionId": "cs1", "ice": {"candidate": "c"}}
    ]


def test_local_end_session_forwards_to_central_and_releases_lock() -> None:
    """Local `endSession` clears maps, forwards to central, releases the lock."""
    lock = _FakeLock()
    relay, central, _local = _make_relay_with_journals(robot_app_lock=lock)
    relay._session_to_local_peer["cs1"] = "consumer"
    relay._central_to_local_session["cs1"] = "ls1"
    relay._local_to_central_session["ls1"] = "cs1"

    asyncio.run(
        relay._process_local_message({"type": "endSession", "sessionId": "ls1"})
    )

    assert relay._local_to_central_session == {}
    assert relay._central_to_local_session == {}
    assert central.sent == [{"type": "endSession", "sessionId": "cs1"}]
    assert lock.released == 1
    assert lock.handoff_releases == 1


# ---------------------------------------------------------------------------
# State transitions & status
# ---------------------------------------------------------------------------


def test_set_state_notifies_callback_and_swallows_its_errors() -> None:
    """State changes fire the callback; a raising callback is swallowed."""
    seen: list[tuple[RelayState, Any]] = []
    relay = _make_relay()
    relay._on_state_change = lambda state, msg: seen.append((state, msg))

    relay._set_state(RelayState.CONNECTING, "starting")
    assert relay.state == RelayState.CONNECTING
    assert relay.state_message == "starting"
    assert seen == [(RelayState.CONNECTING, "starting")]

    # A no-op transition (same state + message) does not re-fire.
    relay._set_state(RelayState.CONNECTING, "starting")
    assert len(seen) == 1

    def _boom(state: RelayState, msg: Any) -> None:
        raise RuntimeError("callback blew up")

    relay._on_state_change = _boom
    relay._set_state(RelayState.ERROR, "bad")  # must not raise
    assert relay.state == RelayState.ERROR


def test_get_relay_status_reports_stopped_without_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no relay instance, status is STOPPED / not connected."""
    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "_relay_instance", None)
    status = relay_module.get_relay_status()
    assert status["state"] == RelayState.STOPPED.value
    assert status["is_connected"] is False


def test_get_relay_status_reports_connected_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CONNECTED relay singleton is reflected in the status dict."""
    import reachy_mini.media.central_signaling_relay as relay_module

    relay = _make_relay()
    relay._set_state(RelayState.CONNECTED, "up")
    monkeypatch.setattr(relay_module, "_relay_instance", relay)

    status = relay_module.get_relay_status()
    assert status["state"] == RelayState.CONNECTED.value
    assert status["message"] == "up"
    assert status["is_connected"] is True


# ---------------------------------------------------------------------------
# Send methods, connection teardown, module singletons
# ---------------------------------------------------------------------------
#
# The two send legs and the teardown/singleton helpers are driven directly with
# stub aiohttp/websocket objects — no real sockets or HTTP — to cover the wire
# emission, error swallowing, and lifecycle bookkeeping.


class _FakeResponse:
    """aiohttp response stand-in usable as an async context manager."""

    def __init__(self, status: int) -> None:
        self.status = status

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def text(self) -> str:
        return "body"


class _FakeHTTPSession:
    """aiohttp ClientSession stand-in recording POSTs."""

    def __init__(self, status: int = 200) -> None:
        self._status = status
        self.posts: list[tuple[str, Any, Any]] = []
        self.timeouts: list[Any] = []
        self.closed = False

    def post(
        self,
        url: str,
        json: Any = None,
        headers: Any = None,
        allow_redirects: bool = True,
        proxy: Any = None,
        timeout: Any = None,
    ) -> _FakeResponse:
        self.posts.append((url, json, headers))
        self.timeouts.append(timeout)
        return _FakeResponse(self._status)

    def get(self, url: str, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(self._status)

    async def close(self) -> None:
        self.closed = True


class _FakeWebsocket:
    """Local-GStreamer websocket stand-in recording sends."""

    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[str] = []
        self._fail = fail
        self.closed = False

    async def send(self, message: str) -> None:
        if self._fail:
            raise RuntimeError("ws send boom")
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True


def test_send_to_central_skips_without_session_or_token() -> None:
    """No HTTP session or no token -> the send is a no-op."""
    relay = _make_relay()
    relay._http_session = None
    relay.hf_token = "tok"
    asyncio.run(relay._send_to_central({"type": "x"}))  # no session

    relay._http_session = _FakeHTTPSession()  # type: ignore[assignment]
    relay.hf_token = None
    asyncio.run(relay._send_to_central({"type": "x"}))  # no token
    assert relay._http_session.posts == []


def test_send_to_central_posts_with_bearer_token() -> None:
    """A send POSTs to <central>/send with the token in the Authorization header."""
    relay = _make_relay()
    relay._http_session = _FakeHTTPSession(status=200)  # type: ignore[assignment]
    relay.hf_token = "tok"
    asyncio.run(relay._send_to_central({"type": "peer"}))
    url, body, headers = relay._http_session.posts[0]
    assert url == f"{relay.central_uri}/send"
    assert body == {"type": "peer"}
    assert headers == {"Authorization": "Bearer tok"}


def test_send_to_central_bounds_each_request() -> None:
    """Every POST carries its own budget, not the session's 300 s default."""
    import reachy_mini.media.central_signaling_relay as relay_module

    relay = _make_relay()
    relay._http_session = _FakeHTTPSession(status=200)  # type: ignore[assignment]
    relay.hf_token = "tok"
    asyncio.run(relay._send_to_central({"type": "setPeerStatus"}))
    assert relay._http_session.timeouts[0].total == relay_module.CENTRAL_SEND_TIMEOUT_S


def test_send_to_central_fails_fast_when_central_goes_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that accepts and never answers (packets dropped) is a failure.

    This is what a heartbeat sees when the network stalls under a
    kept-alive connection; without a budget the POST hangs and the
    heartbeat never counts a failure.
    """
    import aiohttp

    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "CENTRAL_SEND_TIMEOUT_S", 0.3)

    async def run() -> tuple[bool, float]:
        async def swallow(
            reader: asyncio.StreamReader, _w: asyncio.StreamWriter
        ) -> None:
            await reader.read(-1)  # read the request, never reply

        server = await asyncio.start_server(swallow, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        relay = _make_relay()
        relay.central_uri = f"http://127.0.0.1:{port}"
        relay.hf_token = "tok"
        relay._http_session = aiohttp.ClientSession()
        try:
            start = time.monotonic()
            ok = await relay._send_to_central({"type": "setPeerStatus"})
            return ok, time.monotonic() - start
        finally:
            await relay._http_session.close()
            server.close()

    ok, elapsed = asyncio.run(run())
    assert ok is False
    assert elapsed < 2.0


def test_send_to_central_non_200_does_not_raise() -> None:
    """A non-200 response is logged, not raised."""
    relay = _make_relay()
    relay._http_session = _FakeHTTPSession(status=503)  # type: ignore[assignment]
    relay.hf_token = "tok"
    asyncio.run(relay._send_to_central({"type": "x"}))
    assert len(relay._http_session.posts) == 1


def test_send_to_local_sends_json_over_ws() -> None:
    """A local send serialises the message and writes it to the websocket."""
    relay = _make_relay()
    relay._local_ws = _FakeWebsocket()  # type: ignore[assignment]
    asyncio.run(relay._send_to_local({"type": "list"}))
    assert relay._local_ws.sent == [json.dumps({"type": "list"})]


def test_send_to_local_no_ws_is_noop() -> None:
    """No local websocket -> the send is a no-op."""
    relay = _make_relay()
    relay._local_ws = None
    asyncio.run(relay._send_to_local({"type": "list"}))  # no raise


def test_send_to_local_swallows_send_errors() -> None:
    """A websocket send failure is caught, not propagated."""
    relay = _make_relay()
    relay._local_ws = _FakeWebsocket(fail=True)  # type: ignore[assignment]
    asyncio.run(relay._send_to_local({"type": "list"}))  # no raise


def test_tear_down_active_sessions_notifies_both_and_clears() -> None:
    """Teardown ends every session on both legs and clears all bookkeeping."""
    relay, central, local = _make_relay_with_journals()
    relay._central_to_local_session["cs1"] = "ls1"
    relay._local_to_central_session["ls1"] = "cs1"
    relay._session_to_local_peer["cs1"] = "consumer"
    relay._pending_central_sessions.append("cs2")

    asyncio.run(relay._tear_down_active_sessions(reason="local_app_started"))

    assert central.sent == [
        {"type": "endSession", "sessionId": "cs1", "reason": "local_app_started"}
    ]
    assert local.sent == [{"type": "endSession", "sessionId": "ls1"}]
    assert relay._central_to_local_session == {}
    assert relay._local_to_central_session == {}
    assert relay._session_to_local_peer == {}
    assert relay._pending_central_sessions == []


class _RunningLoop:
    """An event loop running in a thread, standing in for the relay's own."""

    def __enter__(self) -> asyncio.AbstractEventLoop:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        return self.loop

    def __exit__(self, *_exc: Any) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2.0)
        self.loop.close()


def test_stop_ends_sessions_on_central_first() -> None:
    """A stopping relay tells central, with a reason, that its sessions are over.

    Central keeps a session across the robot's disconnect, so without this
    the client only learns from its own link timeouts.
    """
    relay, central, local = _make_relay_with_journals()
    relay._central_to_local_session["cs1"] = "ls1"
    relay._local_to_central_session["ls1"] = "cs1"
    with _RunningLoop() as loop:
        relay._thread_loop = loop
        asyncio.run(relay.stop())

    assert central.sent == [
        {"type": "endSession", "sessionId": "cs1", "reason": "robot_stopping"}
    ]
    assert local.sent == [{"type": "endSession", "sessionId": "ls1"}]


def test_stop_does_not_wait_on_an_unreachable_central(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ending the sessions is bounded: a hung central cannot hold up a stop."""
    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "STOP_END_SESSIONS_TIMEOUT_S", 0.2)
    relay, _central, _local = _make_relay_with_journals()

    async def hang(_msg: dict[str, Any]) -> bool:
        await asyncio.sleep(30)
        return True

    relay._send_to_central = hang  # type: ignore[method-assign]
    relay._central_to_local_session["cs1"] = "ls1"
    relay._local_to_central_session["ls1"] = "cs1"
    with _RunningLoop() as loop:
        relay._thread_loop = loop
        start = time.monotonic()
        asyncio.run(relay.stop())
        elapsed = time.monotonic() - start

    assert elapsed < 2.0
    assert relay.state == RelayState.STOPPED


def test_remote_eviction_never_hangs_the_local_app_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tear-down that never completes cannot hold up a local app's start.

    If the relay loop exits before running it, the eviction future never
    resolves: the app start, already holding the lock, used to hang there
    and every later start was refused until a daemon restart.
    """
    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "EVICTION_TIMEOUT_S", 0.2)
    relay = _make_relay()

    async def never_finishes(reason: str, **_kw: Any) -> None:
        await asyncio.sleep(30)

    relay._tear_down_active_sessions = never_finishes  # type: ignore[method-assign]
    with _RunningLoop() as loop:
        relay._thread_loop = loop
        start = time.monotonic()
        asyncio.run(relay._handle_remote_eviction())
        elapsed = time.monotonic() - start
    assert elapsed < 2.0


def test_a_slow_remote_eviction_still_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the bound the app start stops waiting, but the tear-down finishes.

    Cancelling it instead could leave the remote session half torn down
    (central told, the local peer connection never closed) next to the app.
    """
    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "EVICTION_TIMEOUT_S", 0.1)
    relay = _make_relay()
    finished = threading.Event()

    async def slow(reason: str, **_kw: Any) -> None:
        await asyncio.sleep(0.4)
        finished.set()

    relay._tear_down_active_sessions = slow  # type: ignore[method-assign]
    with _RunningLoop() as loop:
        relay._thread_loop = loop
        asyncio.run(relay._handle_remote_eviction())
        assert finished.wait(timeout=2.0)


def test_stop_without_sessions_sends_nothing() -> None:
    """No session, nothing to tell central."""
    relay, central, _local = _make_relay_with_journals()
    with _RunningLoop() as loop:
        relay._thread_loop = loop
        asyncio.run(relay.stop())
    assert central.sent == []


def test_close_connections_closes_and_clears() -> None:
    """Closing drops the session/ws, clears session state, releases the lock."""
    lock = _FakeLock()
    relay = _make_relay()
    relay._robot_app_lock = lock  # type: ignore[assignment]
    relay._http_session = _FakeHTTPSession()  # type: ignore[assignment]
    relay._local_ws = _FakeWebsocket()  # type: ignore[assignment]
    relay._central_peer_id = "C1"
    relay._central_to_local_session["cs1"] = "ls1"
    session, ws = relay._http_session, relay._local_ws

    asyncio.run(relay._close_connections())

    assert session.closed is True and ws.closed is True
    assert relay._http_session is None and relay._local_ws is None
    assert relay._central_peer_id is None
    assert relay._central_to_local_session == {}
    assert lock.released == 1


def test_close_connections_reentrant_guard_short_circuits() -> None:
    """A concurrent teardown (``_closing`` set) is a no-op."""
    relay = _make_relay()
    relay._closing = True
    session = _FakeHTTPSession()
    relay._http_session = session  # type: ignore[assignment]

    asyncio.run(relay._close_connections())

    assert session.closed is False
    assert relay._http_session is session


# ---- module-level singletons ----


def test_get_relay_returns_the_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_relay` exposes the module singleton (or None)."""
    import reachy_mini.media.central_signaling_relay as m

    monkeypatch.setattr(m, "_relay_instance", None)
    assert m.get_relay() is None
    sentinel = _make_relay()
    monkeypatch.setattr(m, "_relay_instance", sentinel)
    assert m.get_relay() is sentinel


def test_start_central_relay_returns_existing_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second start returns the running relay without creating another."""
    import reachy_mini.media.central_signaling_relay as m

    existing = _make_relay()
    monkeypatch.setattr(m, "_relay_instance", existing)
    assert asyncio.run(m.start_central_relay()) is existing


def test_stop_central_relay_stops_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopping awaits the relay's stop and clears the singleton."""
    import reachy_mini.media.central_signaling_relay as m

    class _FakeRelay:
        def __init__(self) -> None:
            self.stopped = False

        async def stop(self) -> None:
            self.stopped = True

    fake = _FakeRelay()
    monkeypatch.setattr(m, "_relay_instance", fake)
    asyncio.run(m.stop_central_relay())
    assert fake.stopped is True
    assert m._relay_instance is None


def test_notify_token_change_noop_without_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token change with no relay running is a clean no-op."""
    import reachy_mini.media.central_signaling_relay as m

    monkeypatch.setattr(m, "_relay_instance", None)
    asyncio.run(m.notify_token_change("tok"))  # no raise


def test_notify_token_change_forwards_to_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token change is forwarded to the running relay's update_token."""
    import reachy_mini.media.central_signaling_relay as m

    class _FakeRelay:
        def __init__(self) -> None:
            self.token: Any = "unset"

        async def update_token(self, token: Any) -> None:
            self.token = token

    fake = _FakeRelay()
    monkeypatch.setattr(m, "_relay_instance", fake)
    asyncio.run(m.notify_token_change("newtok"))
    assert fake.token == "newtok"


def test_relay_recovers_from_an_unauthorized_sse_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 401 mid-session refreshes the credentials and retries without a teardown."""
    from reachy_mini.apps.sources import hf_auth

    relay = _make_relay()
    relay.hf_token = "stale"
    relay._http_session = _FakeHTTPSession(status=401)  # type: ignore[assignment]
    monkeypatch.setattr(
        hf_auth,
        "get_hf_token",
        lambda force_refresh=False: "fresh",
    )

    asyncio.run(relay._handle_central_sse())

    assert relay.hf_token == "fresh"
    assert not relay._token_updated.is_set()
    assert relay._connection_attempts == 1
    assert relay.state is not RelayState.ERROR


def test_relay_reports_a_401_it_cannot_recover_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecoverable 401 surfaces as ERROR rather than a silent reconnect loop."""
    from reachy_mini.apps.sources import hf_auth

    relay = _make_relay()
    relay.hf_token = "stale"
    relay._http_session = _FakeHTTPSession(status=401)  # type: ignore[assignment]
    monkeypatch.setattr(
        hf_auth,
        "get_hf_token",
        lambda force_refresh=False: "stale",
    )

    asyncio.run(relay._handle_central_sse())

    assert relay.state is RelayState.ERROR


def test_relay_backs_off_when_central_rejects_every_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every 401 refreshes, but only the first of a session retries at once."""
    from reachy_mini.apps.sources import hf_auth

    relay = _make_relay()
    relay.hf_token = "stale"
    relay._http_session = _FakeHTTPSession(status=401)  # type: ignore[assignment]
    refreshes: list[bool] = []

    def _always_new_token(force_refresh: bool = False) -> str:
        refreshes.append(force_refresh)
        return f"fresh-{len(refreshes)}"

    monkeypatch.setattr(hf_auth, "get_hf_token", _always_new_token)

    asyncio.run(relay._handle_central_sse())
    assert relay.state is not RelayState.ERROR

    asyncio.run(relay._handle_central_sse())

    assert refreshes == [True, True]
    assert relay.state is RelayState.ERROR


def test_run_loop_retries_at_once_after_a_refreshed_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The short-attempt backoff must not delay the first 401's retry."""
    from reachy_mini.apps.sources import hf_auth

    relay = _make_relay()
    relay._running = True
    relay.hf_token = "expired"
    monkeypatch.setattr(hf_auth, "get_hf_token", lambda force_refresh=False: "fresh")
    calls = 0

    async def fake_connect_and_relay() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            relay._http_session = _FakeHTTPSession(status=401)  # type: ignore[assignment]
            await relay._handle_central_sse()
        else:
            relay._running = False

    relay._connect_and_relay = fake_connect_and_relay  # type: ignore[method-assign]

    start = time.monotonic()
    asyncio.run(relay._run_loop())

    assert calls == 2
    assert time.monotonic() - start < 1.0


def test_connect_skipped_when_stop_raced_the_attempt() -> None:
    """stop()'s wake-up is cleared by a starting attempt: it must not connect."""
    relay = _make_relay()
    relay._running = False
    relay._token_updated.set()
    refreshed: list[bool] = []
    relay._refresh_token = lambda force_refresh=False: refreshed.append(True)  # type: ignore[method-assign,func-returns-value]

    asyncio.run(relay._connect_and_relay())

    assert refreshed == []


def test_relay_recovers_after_a_failed_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refresh that failed on one 401 is retried on the next, so the relay recovers."""
    from reachy_mini.apps.sources import hf_auth

    relay = _make_relay()
    relay.hf_token = "stale"
    relay._http_session = _FakeHTTPSession(status=401)  # type: ignore[assignment]
    tokens = iter(["stale", "fresh"])
    monkeypatch.setattr(
        hf_auth, "get_hf_token", lambda force_refresh=False: next(tokens)
    )

    asyncio.run(relay._handle_central_sse())
    assert relay.state is RelayState.ERROR

    asyncio.run(relay._handle_central_sse())

    assert relay.hf_token == "fresh"


# ---------------------------------------------------------------------------
# Central-only drop: established sessions survive peer-to-peer
# ---------------------------------------------------------------------------


def _relay_with_live_session(
    lock: RobotAppLock,
) -> tuple[CentralSignalingRelay, _SendJournal, _SendJournal]:
    """Relay holding one established session (s1 <-> local-1) and the lock."""
    relay, central, local = _relay_with_lock(lock)
    _start_session(relay)
    asyncio.run(
        relay._process_local_message({"type": "sessionStarted", "sessionId": "local-1"})
    )
    local.sent.clear()
    central.sent.clear()
    return relay, central, local


def test_central_loss_keeps_established_sessions_and_the_lock() -> None:
    """Losing central doesn't end a live session or free the robot."""
    lock = RobotAppLock()
    relay, _central, local = _relay_with_live_session(lock)
    relay._pending_central_sessions.append("s2")
    relay._session_to_local_peer["s2"] = "p2"

    relay._keep_sessions_on_central_loss()

    assert relay._local_to_central_session == {"local-1": "s1"}
    assert relay._orphaned_local_sessions == {"local-1"}
    assert relay._pending_central_sessions == []
    assert "s2" not in relay._session_to_local_peer
    assert local.sent == []  # nothing ended locally
    assert lock.status().state == RobotAppLockState.REMOTE_SESSION


def test_central_loss_with_only_pending_sessions_frees_the_robot() -> None:
    """A session that never got past central can't complete: drop it and its lock."""
    lock = RobotAppLock()
    relay, _central, _local = _relay_with_lock(lock)
    _start_session(relay)  # pending only, lock taken

    relay._keep_sessions_on_central_loss()

    assert relay._pending_central_sessions == []
    assert lock.status().state == RobotAppLockState.FREE


def test_new_session_supersedes_orphaned_ones() -> None:
    """A fresh request (e.g. the phone re-dialing) replaces a kept session."""
    lock = RobotAppLock()
    relay, central, local = _relay_with_live_session(lock)
    relay._keep_sessions_on_central_loss()

    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "p", "sessionId": "s9"}
        )
    )

    assert {"type": "endSession", "sessionId": "local-1"} in local.sent
    assert relay._orphaned_local_sessions == set()
    assert "local-1" not in relay._local_to_central_session
    assert relay._pending_central_sessions == ["s9"]
    assert not any(m.get("reason", "").startswith("robot_busy") for m in central.sent)
    assert lock.status().state == RobotAppLockState.REMOTE_SESSION


def test_orphan_ending_locally_frees_the_robot() -> None:
    """When WebRTC drops the kept session (peer gone), the robot is released."""
    lock = RobotAppLock()
    relay, _central, _local = _relay_with_live_session(lock)
    relay._keep_sessions_on_central_loss()

    asyncio.run(
        relay._process_local_message({"type": "endSession", "sessionId": "local-1"})
    )

    assert relay._orphaned_local_sessions == set()
    assert relay._local_to_central_session == {}
    assert lock.status().state == RobotAppLockState.FREE


# ---------------------------------------------------------------------------
# Liveness judged on the peer-to-peer link (data channel), not on central
# ---------------------------------------------------------------------------


def _settled_live_session(
    lock: RobotAppLock, idle_s: Optional[float]
) -> tuple[CentralSignalingRelay, _SendJournal, _SendJournal]:
    """Build an established, settled session whose link probe reports ``idle_s``."""
    relay, central, local = _relay_with_live_session(lock)
    relay._local_session_started_at["local-1"] = time.monotonic() - 60.0
    relay.set_session_idle_probe(lambda _sid: idle_s)
    return relay, central, local


def test_central_end_without_reason_keeps_a_live_session() -> None:
    """Central evicting the client (no reason) doesn't end a session whose link is live."""
    lock = RobotAppLock()
    relay, _central, local = _settled_live_session(lock, idle_s=0.4)

    asyncio.run(
        relay._process_central_message({"type": "endSession", "sessionId": "s1"})
    )

    assert local.sent == []
    assert relay._local_to_central_session == {"local-1": "s1"}
    assert relay._orphaned_local_sessions == {"local-1"}
    assert lock.status().state == RobotAppLockState.REMOTE_SESSION


def test_central_end_with_reason_is_honoured() -> None:
    """A deliberate end (reason given) still ends the session locally."""
    lock = RobotAppLock()
    relay, _central, local = _settled_live_session(lock, idle_s=0.4)

    asyncio.run(
        relay._process_central_message(
            {"type": "endSession", "sessionId": "s1", "reason": "client_stop"}
        )
    )

    assert {"type": "endSession", "sessionId": "local-1"} in local.sent
    assert relay._local_to_central_session == {}


def test_central_end_without_reason_ends_an_idle_session() -> None:
    """With a quiet link the eviction is believed, as before."""
    lock = RobotAppLock()
    relay, _central, local = _settled_live_session(lock, idle_s=30.0)

    asyncio.run(
        relay._process_central_message({"type": "endSession", "sessionId": "s1"})
    )

    assert {"type": "endSession", "sessionId": "local-1"} in local.sent


def test_new_request_is_refused_while_the_link_is_live() -> None:
    """Another device can't take the robot from a session that is in use."""
    lock = RobotAppLock()
    relay, central, local = _settled_live_session(lock, idle_s=0.4)

    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "p2", "sessionId": "s9"}
        )
    )

    assert {
        "type": "endSession",
        "sessionId": "s9",
        "reason": "robot_busy_local",
    } in central.sent
    assert relay._local_to_central_session == {"local-1": "s1"}
    assert {"type": "endSession", "sessionId": "local-1"} not in local.sent


def test_new_request_replaces_a_ghost_session() -> None:
    """A settled session whose data channel went quiet is superseded."""
    lock = RobotAppLock()
    relay, central, local = _settled_live_session(lock, idle_s=30.0)

    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "p2", "sessionId": "s9"}
        )
    )

    assert {"type": "endSession", "sessionId": "local-1"} in local.sent
    assert relay._pending_central_sessions == ["s9"]
    assert not any(m.get("reason", "").startswith("robot_busy") for m in central.sent)


def test_a_young_session_is_never_taken_as_a_ghost() -> None:
    """A session still negotiating (no data channel yet) is protected."""
    lock = RobotAppLock()
    relay, central, _local = _relay_with_live_session(lock)
    # Quiet so far, but mapped moments ago: still negotiating.
    relay.set_session_idle_probe(lambda _sid: 30.0)

    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "p2", "sessionId": "s9"}
        )
    )

    assert {
        "type": "endSession",
        "sessionId": "s9",
        "reason": "robot_busy_local",
    } in central.sent
    assert relay._local_to_central_session == {"local-1": "s1"}


def test_meta_reports_session_liveness_and_activity() -> None:
    """Central keeps our sessions across SSE drops and can list us busy."""
    relay = _make_relay()
    meta = relay._build_producer_meta()
    assert meta["session_liveness"] == "peer"
    assert meta["session_active"] is False

    relay._local_to_central_session["local-1"] = "s1"
    assert relay._build_producer_meta()["session_active"] is True


def test_ghost_takeover_tells_central_the_old_session_was_superseded() -> None:
    """Central (and the old client) learn the ghost was replaced."""
    lock = RobotAppLock()
    relay, central, _local = _settled_live_session(lock, idle_s=30.0)

    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "p2", "sessionId": "s9"}
        )
    )

    assert {
        "type": "endSession",
        "sessionId": "s1",
        "reason": "superseded",
    } in central.sent


def test_one_live_session_keeps_the_robot_even_beside_a_ghost() -> None:
    """Liveness is per session: a live one blocks a takeover of the others."""
    lock = RobotAppLock()
    relay, central, local = _settled_live_session(lock, idle_s=0.0)
    relay._local_to_central_session["local-2"] = "s2"
    relay._central_to_local_session["s2"] = "local-2"
    relay._local_session_started_at["local-2"] = time.monotonic() - 60.0
    relay.set_session_idle_probe(lambda sid: 0.5 if sid == "local-1" else 30.0)

    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "p3", "sessionId": "s9"}
        )
    )

    assert local.sent == []
    assert set(relay._local_to_central_session) == {"local-1", "local-2"}
    assert {
        "type": "endSession",
        "sessionId": "s9",
        "reason": "robot_busy_local",
    } in central.sent


def test_an_unjudgeable_session_is_never_taken_as_a_ghost() -> None:
    """A session the media server doesn't track (probe None) keeps the robot."""
    lock = RobotAppLock()
    relay, central, local = _settled_live_session(lock, idle_s=None)

    asyncio.run(
        relay._process_central_message(
            {"type": "startSession", "peerId": "p2", "sessionId": "s9"}
        )
    )

    assert local.sent == []
    assert {
        "type": "endSession",
        "sessionId": "s9",
        "reason": "robot_busy_local",
    } in central.sent


def test_failed_heartbeats_end_the_central_leg() -> None:
    """Two failed heartbeats in a row return, triggering a central-only reconnect."""
    relay = _make_relay()
    relay._running = True
    relay._state = RelayState.CONNECTED
    relay._heartbeat_interval_seconds = 0.0
    sent: list[dict[str, Any]] = []

    async def failing_send(msg: dict[str, Any]) -> bool:
        sent.append(msg)
        return False

    relay._send_to_central = failing_send  # type: ignore[method-assign]

    asyncio.run(asyncio.wait_for(relay._heartbeat_loop(), timeout=2.0))

    assert len(sent) == 2


def test_close_connections_cancels_the_kept_local_handler() -> None:
    """The local handler kept across central drops doesn't leak past a close."""

    async def scenario() -> asyncio.Task[None]:
        relay = _make_relay()
        relay._local_task = asyncio.create_task(asyncio.sleep(3600))
        task = relay._local_task
        await relay._close_connections()
        return task

    task = asyncio.run(scenario())
    assert task.cancelled()


def test_rebuilt_local_websocket_drops_stale_sessions_and_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sessions mapped on a dead local websocket are forgotten on reconnect."""
    import reachy_mini.media.central_signaling_relay as relay_module

    lock = RobotAppLock()
    relay, _central, _local = _relay_with_live_session(lock)

    class _FakeHTTP:
        async def close(self) -> None:
            return None

    relay._http_session = _FakeHTTP()  # type: ignore[assignment]

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise OSError("local signalling down")

    monkeypatch.setattr(relay_module.websockets, "connect", refuse)

    with pytest.raises(OSError):
        asyncio.run(relay._connect_local())

    assert relay._local_to_central_session == {}
    assert lock.status().state == RobotAppLockState.FREE
    assert relay._build_producer_meta()["session_active"] is False
