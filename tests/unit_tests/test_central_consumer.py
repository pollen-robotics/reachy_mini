"""Tests for the cloud-backend consumer's outbound command path.

`send_command` is the only part of `ReachyCentralConsumer` that runs
without a peer connection: it serializes an envelope and marshals the
send onto aiortc's loop. Both are faked here, so what is under test is
the wire format the daemon will have to parse.
"""

import json

from reachy_mini.io.protocol import GotoTargetCmd, SetFullTargetCmd
from reachy_mini.media.central_consumer import ReachyCentralConsumer


class _FakeChannel:
    """Records what would go out on the RTCDataChannel."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, payload: str) -> None:
        self.sent.append(payload)


class _InlineLoop:
    """Runs the marshalled callback immediately instead of on aiortc's loop."""

    def call_soon_threadsafe(self, fn) -> None:  # type: ignore[no-untyped-def]
        fn()


def _wired() -> tuple[ReachyCentralConsumer, _FakeChannel]:
    """Build a consumer with an open, fake command channel."""
    consumer = ReachyCentralConsumer(hf_token="hf_test")
    channel = _FakeChannel()
    consumer._cmd_channel = channel  # type: ignore[assignment]
    consumer._cmd_channel_open = True
    consumer._cmd_loop = _InlineLoop()  # type: ignore[assignment]
    return consumer, channel


def test_send_command_serializes_a_protocol_model() -> None:
    """A command model goes out as the `{"type": ...}` envelope daemon-side."""
    consumer, channel = _wired()

    assert consumer.send_command(GotoTargetCmd(head=[1.0, 2.0], duration=0.4)) is True

    assert json.loads(channel.sent[0]) == {
        "type": "goto_target",
        "head": [1.0, 2.0],
        "duration": 0.4,
    }


def test_send_command_omits_unset_optional_fields() -> None:
    """`exclude_none` keeps unset fields off the wire at control rate.

    Every nullable field in the command union defaults to None, so the
    daemon parses this identically to one carrying explicit nulls.
    """
    consumer, channel = _wired()

    consumer.send_command(SetFullTargetCmd(head=[1.0]))

    payload = json.loads(channel.sent[0])
    assert payload == {"type": "set_full_target", "head": [1.0]}
    assert "antennas" not in payload
    assert "body_yaw" not in payload


def test_send_command_passes_dicts_through_unchanged() -> None:
    """JSON-RPC frames share this channel and are not commands."""
    consumer, channel = _wired()
    frame = {"jsonrpc": "2.0", "id": 1, "method": "apps.list", "params": {}}

    assert consumer.send_command(frame) is True

    assert json.loads(channel.sent[0]) == frame


def test_send_command_refuses_when_channel_closed() -> None:
    """No open channel means False and nothing queued, not an exception."""
    consumer, channel = _wired()
    consumer._cmd_channel_open = False

    assert consumer.send_command(GotoTargetCmd(duration=0.1)) is False
    assert channel.sent == []
