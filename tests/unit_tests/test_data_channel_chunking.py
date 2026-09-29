"""Messages too large for the WebRTC data channel are split, not dropped."""

import json

from reachy_mini.media.webrtc_utils import (
    DATA_CHANNEL_MAX_MESSAGE_BYTES,
    split_for_data_channel,
)

# The limit the data channel actually drops at, measured on a robot.
CHANNEL_LIMIT = 65_536


def _reassemble(frames: list[str]) -> str:
    chunks = [json.loads(frame) for frame in frames]
    assert {c["type"] for c in chunks} == {"message_chunk"}
    assert len({c["id"] for c in chunks}) == 1
    assert [c["index"] for c in chunks] == list(range(chunks[0]["count"]))
    return "".join(c["data"] for c in chunks)


def test_a_message_that_fits_goes_out_unchanged() -> None:
    """Leave today's traffic byte-identical."""
    message = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "x" * 50_000})
    assert split_for_data_channel(message) == [message]


def test_a_large_reply_round_trips_in_frames_the_channel_delivers() -> None:
    """Split a relayed `personalities.avatar`-sized reply and join it back."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        + '<path d="M0 0"/>' * 10_000
        + "</svg>"
    )
    message = json.dumps({"jsonrpc": "2.0", "id": "sorry_bro", "result": {"svg": svg}})
    assert len(message.encode()) > DATA_CHANNEL_MAX_MESSAGE_BYTES

    frames = split_for_data_channel(message)

    assert len(frames) > 1
    assert all(len(f.encode()) < CHANNEL_LIMIT for f in frames)
    assert _reassemble(frames) == message


def test_worst_case_characters_still_fit() -> None:
    """Keep frames under the limit when every character escapes to 12 bytes."""
    message = "😀" * 30_000
    frames = split_for_data_channel(message)
    assert all(len(f.encode()) < CHANNEL_LIMIT for f in frames)
    assert _reassemble(frames) == message
