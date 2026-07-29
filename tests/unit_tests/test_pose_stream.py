"""Unit tests for the pose-stream command dispatch + state envelope.

Covers the backend side of the live pose stream: ``subscribe_pose`` /
``unsubscribe_pose`` dispatch through :meth:`process_command` (scoped by
``peer_id`` and delegated to the media server), plus the
``build_state_json`` envelope and its monotonic ``seq`` used by the push
provider.

The media-server transport is mocked here; ``GstMediaServer._push_pose``
and friends are covered separately in ``test_media_server_pose.py``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend
from reachy_mini.io.protocol import SubscribePoseCmd, UnsubscribePoseCmd


def _make_backend() -> MockupSimBackend:
    """Build a lightweight backend with no audio, no kinematics warmup."""
    return MockupSimBackend(use_audio=False)


def test_subscribe_pose_delegates_to_media_server() -> None:
    """``subscribe_pose`` scopes the subscription to the peer and acks."""
    backend = _make_backend()
    media = MagicMock()
    backend._media_server = media

    responses: list[dict[str, Any]] = []
    backend.process_command(
        SubscribePoseCmd(), send_response=responses.append, peer_id="peer-1"
    )

    media.set_pose_subscription.assert_called_once_with("peer-1", True)
    assert responses == [{"status": "ok", "command": "subscribe_pose"}]


def test_unsubscribe_pose_delegates_to_media_server() -> None:
    """``unsubscribe_pose`` removes the peer's subscription and acks."""
    backend = _make_backend()
    media = MagicMock()
    backend._media_server = media

    responses: list[dict[str, Any]] = []
    backend.process_command(
        UnsubscribePoseCmd(), send_response=responses.append, peer_id="peer-1"
    )

    media.set_pose_subscription.assert_called_once_with("peer-1", False)
    assert responses == [{"status": "ok", "command": "unsubscribe_pose"}]


def test_subscribe_pose_without_peer_id_is_noop_but_acks() -> None:
    """A transport with no peer identity (e.g. WS) can't scope a subscription.

    The dispatch must not touch the media server, but still ack so a client
    that blindly subscribes on every reconnect doesn't hang waiting.
    """
    backend = _make_backend()
    media = MagicMock()
    backend._media_server = media

    responses: list[dict[str, Any]] = []
    backend.process_command(
        SubscribePoseCmd(), send_response=responses.append, peer_id=None
    )

    media.set_pose_subscription.assert_not_called()
    assert responses == [{"status": "ok", "command": "subscribe_pose"}]


def test_build_state_json_wraps_state_and_increments_seq() -> None:
    """Each frame wraps the state dict and carries a strictly increasing seq."""
    backend = _make_backend()
    backend.build_state_dict = lambda: {"body_yaw": 0.0}  # type: ignore[assignment]

    first = backend.build_state_json()
    second = backend.build_state_json()
    assert first is not None and second is not None

    a = json.loads(first)
    b = json.loads(second)
    assert a["state"] == {"body_yaw": 0.0}
    # Monotonic, strictly increasing so the client can drop stale/out-of-order
    # frames on the unordered pose channel.
    assert a["seq"] == 1
    assert b["seq"] == 2


def test_build_state_json_returns_none_on_build_error() -> None:
    """A transient build failure skips the tick instead of crashing the push."""
    backend = _make_backend()

    def boom() -> dict[str, Any]:
        raise RuntimeError("kinematics not ready")

    backend.build_state_dict = boom  # type: ignore[assignment]
    assert backend.build_state_json() is None
