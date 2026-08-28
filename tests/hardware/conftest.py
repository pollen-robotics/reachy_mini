"""Fixtures for the on-hardware test suite.

These tests need a real Reachy Mini: a real speaker, a real microphone, and the
XMOS XVF3800 audio board reachable over USB from *this* machine.  They are
selected by ``-m "audio and respeaker"`` and are excluded from every normal CI
run (see the marker exclusions in ``.github/workflows/pytest.yml`` and
``allure.yml``, plus ``testpaths`` in ``pyproject.toml``).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator

import pytest
import requests

from reachy_mini.media.audio_control_utils import (
    PARAMETERS,
    AudioConfig,
    ReSpeaker,
    init_respeaker_usb,
)
from reachy_mini.reachy_mini import (
    INIT_ANTENNAS_JOINT_POSITIONS,
    INIT_HEAD_POSE,
    ReachyMini,
)

# Board registers to write while measuring the speaker -> mic path.
#
# PP_ECHOONOFF=0 turns off residual echo suppression, which otherwise removes
# exactly the signal we are trying to measure.  PP_AGCONOFF=0 is not optional
# either: automatic gain control would move the mic gain *during* the recording
# and corrupt any amplitude comparison.
#
# Both are volatile `rw` runtime registers — no firmware write, no daemon
# restart, and a power cycle restores them.  We never write SAVE_CONFIGURATION.
#
# Override during bring-up without editing this file, e.g.
#   REACHY_TEST_AEC_OFF="SHF_BYPASS=1"     # bigger hammer, also kills beamforming
#   REACHY_TEST_AEC_OFF=""                 # negative control: leave AEC on
DEFAULT_AEC_OFF = "PP_ECHOONOFF=0,PP_AGCONOFF=0"


def _parse_aec_config(spec: str) -> AudioConfig:
    """Parse a ``NAME=v[;v...],NAME=v`` spec into an AudioConfig."""
    config: list[tuple[str, list[int]]] = []
    for entry in filter(None, (e.strip() for e in spec.split(","))):
        name, _, values = entry.partition("=")
        if name not in PARAMETERS:
            raise ValueError(f"Unknown audio parameter {name!r} in REACHY_TEST_AEC_OFF")
        config.append((name, [int(v) for v in values.split(";")]))
    return config


@pytest.fixture
def respeaker() -> Iterator[ReSpeaker]:
    """The XVF3800 board over USB, or skip when it isn't on this machine."""
    board = init_respeaker_usb()
    if board is None:
        pytest.skip("No Reachy Mini Audio (XVF3800) USB board on this machine.")
    try:
        yield board
    finally:
        board.close()


@pytest.fixture
def aec_disabled(respeaker: ReSpeaker) -> Iterator[AudioConfig]:
    """Turn off board-side echo cancellation and AGC, restoring them after.

    Yields the config that was applied, so a test can report it.  An empty
    config (``REACHY_TEST_AEC_OFF=""``) is a valid no-op, used as the negative
    control that proves the knob does something.
    """
    wanted = _parse_aec_config(os.environ.get("REACHY_TEST_AEC_OFF", DEFAULT_AEC_OFF))

    # Read every original before writing anything: if we can't read a register
    # we can't put it back, and silently leaving the board altered would poison
    # every later test and every app on this robot.  An empty config is a valid
    # no-op and falls through this whole path unchanged.
    originals: list[tuple[str, list[int]]] = []
    for name, _ in wanted:
        current = respeaker.read_values(name)
        if current is None:
            pytest.fail(f"Could not read {name} back from the board; refusing to write.")
        originals.append((name, [int(v) for v in current]))

    assert respeaker.apply_audio_config(wanted), (
        f"Failed to apply {wanted} to the audio board."
    )
    try:
        yield wanted
    finally:
        # Warn rather than raise: raising here would mask the test's own
        # failure. A warning (not print — teardown stdout is swallowed when
        # the test passes) lands in pytest's warnings summary either way.
        # These registers are volatile, so a power cycle also fixes it.
        if not respeaker.apply_audio_config(originals):
            warnings.warn(
                f"Failed to restore {originals} on the audio board. "
                "Power-cycle the robot before trusting later audio results.",
                stacklevel=2,
            )


@pytest.fixture
def mini() -> Iterator[ReachyMini]:
    """A connected robot. Fails loudly — the daemon must be running."""
    try:
        robot = ReachyMini()
    except Exception as exc:
        # Not a skip: the board fixture already established we're on the right
        # machine, so a missing daemon is a broken setup, and skipping here
        # would read as a pass.
        pytest.fail(
            f"Could not connect to the Reachy Mini daemon: {exc}. "
            "Start it (`reachy-mini-daemon`, or `systemctl start "
            "reachy-mini-daemon` on a Wireless) and retry."
        )
    with robot:
        yield robot


# Pinned levels so the measurement doesn't depend on whatever the user last
# set. Speaker at max for best SNR; mic below max because at 100 the capture
# clips (peak hits 1.0) and clipping harmonics wreck the spectral match.
SPEAKER_VOLUME = 100
MIC_VOLUME = int(os.environ.get("REACHY_TEST_MIC_VOL", "70"))


@pytest.fixture
def pinned_volume() -> Iterator[None]:
    """Pin speaker and mic volume, restoring the user's levels after.

    Uses the daemon's REST API (the suite already requires the daemon on
    localhost: the USB board and the daemon live on the same machine in both
    supported setups).
    """
    base = "http://localhost:8000/api/volume"
    saved: list[tuple[str, int]] = []
    for path, level in (("", SPEAKER_VOLUME), ("/microphone", MIC_VOLUME)):
        resp = requests.get(f"{base}{path}/current", timeout=5)
        resp.raise_for_status()
        saved.append((path, int(resp.json()["volume"])))
        requests.post(
            f"{base}{path}/set", json={"volume": level}, timeout=5
        ).raise_for_status()
    try:
        yield
    finally:
        for path, volume in saved:
            try:
                requests.post(f"{base}{path}/set", json={"volume": volume}, timeout=5)
            except requests.RequestException:
                warnings.warn(
                    f"Failed to restore volume{path or '/speaker'} to {volume}.",
                    stacklevel=2,
                )


@pytest.fixture
def head_raised(mini: ReachyMini) -> Iterator[None]:
    """Raise the head so it isn't sitting over the speaker, muffling it.

    In the sleep pose the head covers the speaker, so a measurement taken there
    reads as a quiet or dead speaker.  This has to be enforced rather than
    documented: otherwise the thresholds get tuned against a blocked speaker.

    Deliberately *not* ``mini.wake_up()`` — that plays ``wake_up.wav`` through
    the very speaker under test, so its chime could be what the mic records,
    passing the test even if the test's own playback did nothing.
    ``goto_target`` is silent and blocking (it waits on task completion).

    No teardown: leaving the head up is harmless, and ``goto_sleep()`` would
    play ``go_sleep.wav`` and cost ~4 s.
    """
    mini.enable_motors()
    mini.goto_target(
        INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS, duration=2.0
    )
    yield
