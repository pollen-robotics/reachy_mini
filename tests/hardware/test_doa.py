"""Direction-of-arrival readout from the XVF3800 board, on real hardware.

Moved from tests/unit_tests/test_audio.py: it needs the physical board, so it
never ran in any automated job there — this suite is where robot-bound tests
get executed.
"""

from __future__ import annotations

import pytest

from reachy_mini.media.audio_control_utils import init_respeaker_usb
from reachy_mini.media.media_manager import MediaBackend, MediaManager


@pytest.mark.audio
@pytest.mark.respeaker
def test_direction_of_arrival() -> None:
    """The board reports an (angle, speech_detected) pair through both APIs."""
    # Gate-and-release rather than the `respeaker` fixture: get_DoA opens its
    # own USB handle to the board, so don't hold a second one open around it.
    board = init_respeaker_usb()
    if board is None:
        pytest.skip("No Reachy Mini Audio (XVF3800) USB board on this machine.")
    board.close()

    media = MediaManager(backend=MediaBackend.LOCAL)
    try:
        # Via GStreamerAudio directly
        doa = media.audio.get_DoA()
        assert doa is not None, "DoA is not defined."
        assert isinstance(doa, tuple), "DoA is not a tuple."
        assert len(doa) == 2, f"DoA has incorrect length: {len(doa)} != 2"
        assert isinstance(doa[0], float), (
            f"DoA has incorrect first type: {type(doa[0])} != float"
        )
        assert isinstance(doa[1], bool), (
            f"DoA has incorrect second type: {type(doa[1])} != bool"
        )
        # Via the MediaManager proxy
        doa_proxy = media.get_DoA()
        assert doa_proxy is not None, "DoA is not defined."
        assert doa_proxy == doa, "Proxy DoA is not equal to direct DoA"
    finally:
        media.close()
