"""Unit tests for WebRTC webrtcbin latency configuration (issue #1407)."""

from __future__ import annotations

import pytest

from reachy_mini.media.webrtc_client_gstreamer import (
    DEFAULT_WEBRTCBIN_LATENCY_MS,
    ENV_WEBRTCBIN_LATENCY_MS,
    resolve_webrtcbin_latency_ms,
)


def test_resolve_default_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default latency is 10 ms when no arg or env is set."""
    monkeypatch.delenv(ENV_WEBRTCBIN_LATENCY_MS, raising=False)
    assert resolve_webrtcbin_latency_ms(None) == DEFAULT_WEBRTCBIN_LATENCY_MS
    assert DEFAULT_WEBRTCBIN_LATENCY_MS == 10


def test_resolve_explicit_latency_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor argument wins over the environment variable."""
    monkeypatch.setenv(ENV_WEBRTCBIN_LATENCY_MS, "200")
    assert resolve_webrtcbin_latency_ms(100) == 100


def test_resolve_env_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variable is used when no explicit argument is given."""
    monkeypatch.setenv(ENV_WEBRTCBIN_LATENCY_MS, "100")
    assert resolve_webrtcbin_latency_ms(None) == 100


def test_resolve_rejects_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative latency values are rejected."""
    with pytest.raises(ValueError, match=">= 0"):
        resolve_webrtcbin_latency_ms(-1)
    monkeypatch.setenv(ENV_WEBRTCBIN_LATENCY_MS, "-5")
    with pytest.raises(ValueError, match=">= 0"):
        resolve_webrtcbin_latency_ms(None)


def test_resolve_rejects_non_integer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-integer environment values are rejected."""
    monkeypatch.setenv(ENV_WEBRTCBIN_LATENCY_MS, "abc")
    with pytest.raises(ValueError, match="integer"):
        resolve_webrtcbin_latency_ms(None)


@pytest.mark.parametrize("value", [2**32, 1.5, True, "invalid"])
def test_resolve_rejects_invalid_explicit_values(value) -> None:
    """Reject overflow and lossy GObject conversions before runtime startup."""
    with pytest.raises(ValueError):
        resolve_webrtcbin_latency_ms(value)


def test_resolve_rejects_env_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment has the same guint limit as an explicit argument."""
    monkeypatch.setenv(ENV_WEBRTCBIN_LATENCY_MS, str(2**32))
    with pytest.raises(ValueError, match="<="):
        resolve_webrtcbin_latency_ms()
