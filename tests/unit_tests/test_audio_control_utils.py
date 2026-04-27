"""Tests for Reachy Mini audio control helpers."""

import struct
from array import array
from unittest.mock import MagicMock

import pytest

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.audio_control_utils import PARAMETERS, ReSpeaker

pytestmark = pytest.mark.audio

CUSTOM_AUDIO_CONFIG = (
    ("PP_MIN_NS", (0.8,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)


class FakeUSBDevice:
    """Minimal USB control-transfer surface used by ReSpeaker tests."""

    def __init__(self) -> None:
        """Initialize the fake parameter storage."""
        self.payloads: dict[tuple[int, int], bytes] = {}
        self.read_overrides: dict[tuple[int, int], bytes] = {}
        self.writes: list[tuple[int, int, bytes]] = []

    def ctrl_transfer(
        self,
        request_type: int,
        request: int,
        wvalue: int,
        windex: int,
        data_or_w_length: object,
        timeout: int,
    ) -> int | array:
        """Record writes and return status-prefixed readback payloads."""
        if isinstance(data_or_w_length, int):
            cmdid = wvalue & 0x7F
            length = data_or_w_length
            payload = self.read_overrides.get(
                (windex, cmdid), self.payloads.get((windex, cmdid), b"")
            )
            payload = payload[: length - 1].ljust(length - 1, b"\x00")
            return array("B", [0, *payload])

        payload = bytes(int(value) & 0xFF for value in data_or_w_length)  # type: ignore[union-attr]
        self.payloads[(windex, wvalue)] = payload
        self.writes.append((windex, wvalue, payload))
        return len(payload)


def _parameter_key(name: str) -> tuple[int, int]:
    parameter = PARAMETERS[name]
    return int(parameter[0]), int(parameter[1])


def test_respeaker_read_values_decodes_numeric_parameters() -> None:
    """Numeric readback should be normalized to tuples without status bytes."""
    respeaker = ReSpeaker(FakeUSBDevice())  # type: ignore[arg-type]

    respeaker.write("PP_MGSCALE", (4.0, 1.0, 1.0))
    assert respeaker.read_values("PP_MGSCALE") == pytest.approx((4.0, 1.0, 1.0))

    respeaker.write("PP_NLATTENONOFF", (0,))
    assert respeaker.read_values("PP_NLATTENONOFF") == (0,)


def test_respeaker_apply_audio_config_writes_and_verifies_custom_config() -> None:
    """Custom config writes should be verified against device readback."""
    device = FakeUSBDevice()
    respeaker = ReSpeaker(device)  # type: ignore[arg-type]

    applied = respeaker.apply_audio_config(CUSTOM_AUDIO_CONFIG, write_settle_seconds=0)

    assert applied is True
    assert [(windex, wvalue) for windex, wvalue, _ in device.writes] == [
        _parameter_key(name) for name, _ in CUSTOM_AUDIO_CONFIG
    ]


def test_respeaker_apply_audio_config_returns_false_when_readback_differs() -> None:
    """Apply should report failure when a write does not stick."""
    device = FakeUSBDevice()
    device.read_overrides[_parameter_key("PP_MGSCALE")] = struct.pack(
        "<fff", 1000.0, 1.0, 1.0
    )
    respeaker = ReSpeaker(device)  # type: ignore[arg-type]

    applied = respeaker.apply_audio_config(
        (("PP_MGSCALE", (4.0, 1.0, 1.0)),),
        write_settle_seconds=0,
    )

    assert applied is False


def test_audio_base_apply_audio_config_uses_respeaker() -> None:
    """Audio backends should use a short-lived ReSpeaker for config writes."""
    config = (("PP_MIN_NS", (0.8,)),)
    respeaker = MagicMock()
    respeaker.apply_audio_config.return_value = True
    audio = MagicMock()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "reachy_mini.media.audio_base.init_respeaker_usb",
            lambda: respeaker,
        )
        applied = AudioBase.apply_audio_config(
            audio, config, verify=False, write_settle_seconds=0
        )

    assert applied is True
    respeaker.apply_audio_config.assert_called_once_with(
        config,
        verify=False,
        write_settle_seconds=0,
    )
    respeaker.close.assert_called_once_with()


def test_audio_base_apply_audio_config_returns_false_without_respeaker() -> None:
    """Audio config application should fail gracefully when no device is found."""
    audio = MagicMock()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "reachy_mini.media.audio_base.init_respeaker_usb",
            lambda: None,
        )
        applied = AudioBase.apply_audio_config(
            audio, (("PP_MIN_NS", (0.8,)),), write_settle_seconds=0
        )

    assert applied is False
    audio.logger.warning.assert_called_once_with("ReSpeaker device not found.")
