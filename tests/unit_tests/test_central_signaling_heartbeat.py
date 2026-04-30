"""Heartbeat re-emission tests for ``CentralSignalingRelay``.

The daemon must keep refreshing its central lease via real
``setPeerStatus`` POSTs, even when the meta payload is identical to the
last one sent. Without this, the central's TTL sweeper would evict an
idle-but-online producer (server-pushed SSE keepalives are NOT a
liveness signal - see ``reachy_mini/docs/SIGNALING.md``).

These tests poke ``update_producer_meta()`` directly without spinning
up the real network stack: we build a relay, force it into
``CONNECTED`` state, replace ``_send_to_central`` with an awaitable
spy, and observe which calls actually go out.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from reachy_mini.daemon.peer_health import PeerHealth
from reachy_mini.media.central_signaling_relay import (
    HEARTBEAT_INTERVAL_SECONDS,
    MAX_HEARTBEAT_INTERVAL_SECONDS,
    MIN_HEARTBEAT_INTERVAL_SECONDS,
    CentralSignalingRelay,
    RelayState,
)


def _make_connected_relay(
    *, health: PeerHealth | None = None
) -> CentralSignalingRelay:
    """Construct a relay short-circuited into ``CONNECTED`` state.

    Real network bring-up touches aiohttp + websockets which we don't
    want in a unit test. Setting the state attributes directly is
    enough for ``update_producer_meta`` whose only liveness gate is
    ``self._state == CONNECTED and self._central_peer_id is not None``.
    """
    relay = CentralSignalingRelay(
        central_uri="https://test.invalid",
        local_uri="ws://127.0.0.1:0",
        hf_token="dummy",
        robot_name="testbot",
        install_id="0" * 32,
        kind="robot",
        wireless_version=True,
        version="0.0.0-test",
        capabilities=["motion"],
        health_provider=(
            lambda: (health or PeerHealth(health="ok", error_code=None))
        ),
    )
    relay._state = RelayState.CONNECTED
    relay._central_peer_id = "fake-peer-id"
    return relay


@pytest.mark.asyncio
async def test_first_call_sends_setpeerstatus():  # noqa: D103
    relay = _make_connected_relay()

    sent: list[dict] = []

    async def _capture(msg):
        sent.append(msg)

    with patch.object(relay, "_send_to_central", side_effect=_capture):
        emitted = await relay.update_producer_meta()

    assert emitted is True
    assert len(sent) == 1
    assert sent[0]["type"] == "setPeerStatus"
    assert sent[0]["roles"] == ["producer"]
    assert relay._last_published_at is not None


@pytest.mark.asyncio
async def test_identical_meta_in_quick_succession_does_not_resend():  # noqa: D103
    relay = _make_connected_relay()

    sent: list[dict] = []

    async def _capture(msg):
        sent.append(msg)

    with patch.object(relay, "_send_to_central", side_effect=_capture):
        await relay.update_producer_meta()  # initial send
        for _ in range(5):
            await relay.update_producer_meta()

    # Only the first call should have hit the wire: subsequent ticks
    # see identical meta and the heartbeat is not yet due.
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_heartbeat_re_emits_after_interval_with_unchanged_meta():
    """Re-emit ``setPeerStatus`` once the heartbeat interval expires.

    Even when the meta payload is byte-for-byte identical, the relay
    MUST re-emit so central can refresh ``last_seen``. This is the
    central's only liveness signal for a steady-state robot.
    """
    relay = _make_connected_relay()

    sent: list[dict] = []

    async def _capture(msg):
        sent.append(msg)

    with patch.object(relay, "_send_to_central", side_effect=_capture):
        await relay.update_producer_meta()
        assert len(sent) == 1

        # Pretend the last send happened a full heartbeat ago. We
        # rewind the cached timestamp instead of sleeping for the
        # full interval - tests must run fast.
        relay._last_published_at = (
            time.monotonic() - HEARTBEAT_INTERVAL_SECONDS - 0.1
        )

        emitted = await relay.update_producer_meta()
        assert emitted is True
        assert len(sent) == 2

        # And the second send carries the same meta as the first.
        assert sent[0]["meta"] == sent[1]["meta"]


@pytest.mark.asyncio
async def test_meta_change_re_emits_immediately_without_waiting_heartbeat():
    """Propagate meta changes on the very next status tick.

    Health going from ``ok`` to ``degraded`` must propagate without
    waiting on the heartbeat interval - users can't be left looking
    at a stale row for 20 s.
    """
    health_box = {"value": PeerHealth(health="ok", error_code=None)}
    relay = CentralSignalingRelay(
        central_uri="https://test.invalid",
        local_uri="ws://127.0.0.1:0",
        hf_token="dummy",
        robot_name="testbot",
        install_id="0" * 32,
        kind="robot",
        wireless_version=True,
        version="0.0.0-test",
        capabilities=["motion"],
        health_provider=lambda: health_box["value"],
    )
    relay._state = RelayState.CONNECTED
    relay._central_peer_id = "fake-peer-id"

    sent: list[dict] = []

    async def _capture(msg):
        sent.append(msg)

    with patch.object(relay, "_send_to_central", side_effect=_capture):
        await relay.update_producer_meta()
        assert len(sent) == 1
        assert sent[0]["meta"]["health"] == "ok"

        # Simulate a transition to ``degraded``. The heartbeat clock
        # is barely advanced, so the only reason to re-emit is the
        # meta delta.
        health_box["value"] = PeerHealth(
            health="degraded", error_code="audio_unavailable"
        )

        emitted = await relay.update_producer_meta()
        assert emitted is True
        assert len(sent) == 2
        assert sent[1]["meta"]["health"] == "degraded"
        assert sent[1]["meta"]["error_code"] == "audio_unavailable"


@pytest.mark.asyncio
async def test_no_send_when_relay_not_connected():
    """Reject sends from non-CONNECTED states.

    A relay still in CONNECTING / CLOSED state has no peer id central
    would associate the message with, so it must short-circuit.
    """
    relay = _make_connected_relay()
    relay._state = RelayState.CONNECTING
    relay._central_peer_id = None

    with patch.object(
        relay, "_send_to_central", side_effect=AssertionError("must not send")
    ):
        emitted = await relay.update_producer_meta()
    assert emitted is False


@pytest.mark.asyncio
async def test_heartbeat_resets_after_send():
    """Restart the heartbeat clock after each emission.

    Otherwise we would keep re-emitting on every subsequent tick once
    the interval was exceeded once.
    """
    relay = _make_connected_relay()

    async def _noop(_msg):
        return None

    with patch.object(relay, "_send_to_central", side_effect=_noop):
        await relay.update_producer_meta()

        relay._last_published_at = (
            time.monotonic() - HEARTBEAT_INTERVAL_SECONDS - 0.1
        )
        await relay.update_producer_meta()  # heartbeat re-emit

        # Next tick: timer was reset, no further send.
        emitted = await relay.update_producer_meta()
        assert emitted is False


# ----------------------------------------------------------------------
# withdraw() resets heartbeat bookkeeping
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Welcome-driven heartbeat negotiation
# ----------------------------------------------------------------------
#
# The relay reads ``recommended_heartbeat_interval_seconds`` from each
# ``welcome`` SSE frame so server operators can re-tune the liveness
# contract by changing one env var on the central, without redeploying
# the fleet. These tests pin the contract so a future refactor that
# accidentally drops the override (or fails to clamp it) breaks loud.


def test_apply_welcome_overrides_heartbeat_interval():
    relay = _make_connected_relay()
    relay._apply_welcome_negotiation(
        {
            "type": "welcome",
            "peerId": "p",
            "lease_seconds": 30.0,
            "recommended_heartbeat_interval_seconds": 10.0,
        }
    )
    assert relay._heartbeat_interval_seconds == 10.0
    assert relay._central_lease_seconds == 30.0


def test_apply_welcome_without_hint_keeps_default():
    relay = _make_connected_relay()
    relay._heartbeat_interval_seconds = 999.0  # poison so we'd notice
    relay._apply_welcome_negotiation(
        {"type": "welcome", "peerId": "p", "username": "alice"}
    )
    assert relay._heartbeat_interval_seconds == HEARTBEAT_INTERVAL_SECONDS
    assert relay._central_lease_seconds is None


def test_apply_welcome_clamps_below_minimum():
    relay = _make_connected_relay()
    relay._apply_welcome_negotiation(
        {"recommended_heartbeat_interval_seconds": 0.1}
    )
    assert relay._heartbeat_interval_seconds == MIN_HEARTBEAT_INTERVAL_SECONDS


def test_apply_welcome_clamps_above_maximum():
    relay = _make_connected_relay()
    relay._apply_welcome_negotiation(
        {"recommended_heartbeat_interval_seconds": 9999.0}
    )
    assert relay._heartbeat_interval_seconds == MAX_HEARTBEAT_INTERVAL_SECONDS


def test_apply_welcome_ignores_non_numeric_hint():
    relay = _make_connected_relay()
    relay._apply_welcome_negotiation(
        {"recommended_heartbeat_interval_seconds": "five"}
    )
    assert relay._heartbeat_interval_seconds == HEARTBEAT_INTERVAL_SECONDS


@pytest.mark.asyncio
async def test_negotiated_heartbeat_drives_re_emit_window():
    """End-to-end: a welcome with a smaller interval must shorten the
    re-emit window seen by ``update_producer_meta``.
    """
    relay = _make_connected_relay()
    relay._apply_welcome_negotiation(
        {"recommended_heartbeat_interval_seconds": 2.0}
    )

    sent: list[dict] = []

    async def _capture(msg):
        sent.append(msg)

    with patch.object(relay, "_send_to_central", side_effect=_capture):
        await relay.update_producer_meta()
        # At t=1.5s (under the 2s negotiated interval), no re-emit.
        relay._last_published_at = time.monotonic() - 1.5
        emitted = await relay.update_producer_meta()
        assert emitted is False
        # At t=2.1s (over), re-emit.
        relay._last_published_at = time.monotonic() - 2.1
        emitted = await relay.update_producer_meta()
        assert emitted is True
    assert len(sent) == 2


@pytest.mark.asyncio
async def test_withdraw_clears_heartbeat_timer():
    """Reset bookkeeping on withdraw so a future re-register sends.

    ``withdraw()`` is followed (later) by a fresh registration. We
    must NOT short-circuit that re-registration's first
    ``setPeerStatus`` because of a stale ``_last_published_meta``.
    """
    relay = _make_connected_relay()

    async def _noop(_msg):
        return None

    with patch.object(relay, "_send_to_central", side_effect=_noop):
        await relay.update_producer_meta()
        assert relay._last_published_meta is not None
        assert relay._last_published_at is not None

        await relay.withdraw(timeout=0.05)

    assert relay._last_published_meta is None
    assert relay._last_published_at is None
