"""Unit tests for :mod:`reachy_mini.media.central_signaling_relay`.

These tests cover only the pure helpers exposed on the relay module:
the welcome-message negotiation logic that picks the heartbeat
cadence. Anything that would require driving the asyncio loop, the
local GStreamer websocket, or the central HTTP server is out of
scope here (and is currently tracked as integration-test tech debt
in the same way as the robot_app_lock test suite).
"""

from __future__ import annotations

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
    result = CentralSignalingRelay._negotiate_heartbeat_interval(
        {"lease_seconds": 0}
    )
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
