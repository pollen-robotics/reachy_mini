"""Fixtures for the on-hardware test suite.

These tests need a real Reachy Mini: a real speaker, a real microphone, and the
XMOS XVF3800 audio board reachable over USB from *this* machine.  They are
selected by ``-m "audio and respeaker"`` and are excluded from every normal CI
run (see the marker exclusions in ``.github/workflows/pytest.yml`` and
``allure.yml``, plus ``testpaths`` in ``pyproject.toml``).
"""

from __future__ import annotations

import os
import time
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
            pytest.fail(
                f"Could not read {name} back from the board; refusing to write."
            )
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


def _pinned_volume(host: str) -> Iterator[None]:
    """Pin speaker and mic volume on ``host``'s daemon, restoring the levels after."""
    base = f"http://{host}:8000/api/volume"
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
def pinned_volume() -> Iterator[None]:
    """Pin volumes on the local daemon (USB board and daemon share a machine)."""
    yield from _pinned_volume("localhost")


# Head height above which the head is considered up, off the speaker. The
# sleep pose sits at z = -45.6 mm and the raised init pose at z ~ 0 (a few mm
# off from calibration residual), so -20 mm splits them with margin both ways.
HEAD_RAISED_MIN_Z_M = -0.020


def _assert_head_raised(z_m: float) -> None:
    """Fail unless the head actually rose; asking it to move is not enough.

    A goto can be accepted and still move nothing (motors disabled, a dead
    write path), and a measurement with the head resting on the speaker reads
    as a muffled or broken speaker.
    """
    if z_m < HEAD_RAISED_MIN_Z_M:
        pytest.fail(
            f"Head is still down (z={z_m * 1000:.1f} mm, need > "
            f"{HEAD_RAISED_MIN_Z_M * 1000:.0f} mm): it would block the speaker. "
            "The move was requested but did not happen; check the motor mode."
        )


@pytest.fixture
def head_raised(mini: ReachyMini) -> Iterator[None]:
    """Raise the head so it isn't sitting over the speaker, muffling it.

    Deliberately *not* ``mini.wake_up()``: that plays ``wake_up.wav`` through
    the very speaker under test. ``goto_target`` is silent and blocking.

    No teardown: leaving the head up is harmless, and ``goto_sleep()`` would
    play ``go_sleep.wav`` and cost ~4 s.
    """
    mini.enable_motors()
    mini.goto_target(
        INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS, duration=2.0
    )
    _assert_head_raised(float(mini.get_current_head_pose()[2, 3]))
    yield


# --- Remote (laptop-as-client) fixtures ------------------------------------
#
# The `wireless`-marked tests run on a machine on the same LAN as a Wireless
# robot and drive it entirely over the network: daemon REST for state, the
# WebRTC media path for audio/video. Nothing here needs USB or SSH.

ROBOT_HOST = os.environ.get("REACHY_TEST_HOST", "reachy-mini.local")


@pytest.fixture(scope="session")
def robot_host() -> str:
    """The robot to test against, or skip when none answers.

    Reachability is the skip gate (same role the USB probe plays for the
    on-robot tests): no robot on the LAN means "not available here", while
    everything after — daemon fails to start, media never flows — is a
    failure.
    """
    try:
        requests.get(f"http://{ROBOT_HOST}:8000/api/daemon/status", timeout=5)
    except requests.RequestException:
        pytest.skip(f"No robot answering at {ROBOT_HOST}:8000 (REACHY_TEST_HOST).")
    return ROBOT_HOST


@pytest.fixture(scope="session")
def daemon_running(robot_host: str) -> str:
    """Ensure the daemon backend is running (starts it if needed).

    The WebRTC producer (webrtcsink + its signalling server on :8443) only
    exists while the backend is running. Raising the head is not this
    fixture's job: ``head_raised_remote`` does it and verifies it. The daemon
    is left running afterwards; it idles fine.
    """
    base = f"http://{robot_host}:8000/api/daemon"
    state = requests.get(f"{base}/status", timeout=5).json().get("state")
    if state != "running":
        # /start is synchronous and includes the wake-up motion, so it can
        # outlive any reasonable request timeout — fire it, tolerate the
        # timeout, and let the poll below be the actual arbiter.
        try:
            requests.post(f"{base}/start", params={"wake_up": "true"}, timeout=60)
        except requests.Timeout:
            pass
        deadline = time.time() + 90
        while state != "running":
            if time.time() > deadline:
                pytest.fail(f"Daemon did not reach 'running' within 90s at {base}.")
            time.sleep(2)
            state = requests.get(f"{base}/status", timeout=5).json().get("state")
        time.sleep(2)  # let the media server finish coming up
    return robot_host


def _remote_apply(base: str, config: AudioConfig) -> bool:
    resp = requests.post(
        f"{base}/api/audio/config/apply",
        json={
            "config": [{"name": n, "values": list(v)} for n, v in config],
            "verify": True,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return bool(resp.json()["applied"])


@pytest.fixture
def aec_disabled_remote(daemon_running: str) -> Iterator[AudioConfig]:
    """The `aec_disabled` fixture, over the daemon's REST API instead of USB.

    Same registers, same save/restore contract; see `aec_disabled` for why
    each parameter is written.
    """
    base = f"http://{daemon_running}:8000"
    wanted = _parse_aec_config(os.environ.get("REACHY_TEST_AEC_OFF", DEFAULT_AEC_OFF))

    originals = [
        (
            name,
            requests.get(
                f"{base}/api/audio/config/parameter/{name}", timeout=10
            ).json()["values"],
        )
        for name, _ in wanted
    ]

    assert _remote_apply(base, wanted), f"Failed to apply {wanted} over REST."
    try:
        yield wanted
    finally:
        if not _remote_apply(base, originals):
            warnings.warn(
                f"Failed to restore {originals} on the audio board (REST). "
                "Power-cycle the robot before trusting later audio results.",
                stacklevel=2,
            )


@pytest.fixture
def pinned_volume_remote(daemon_running: str) -> Iterator[None]:
    """The `pinned_volume` fixture, addressed to a remote robot."""
    yield from _pinned_volume(daemon_running)


def _remote_head_pose(base: str) -> dict[str, float]:
    resp = requests.get(f"{base}/api/state/present_head_pose", timeout=5)
    resp.raise_for_status()
    return {k: float(v) for k, v in resp.json().items()}


def _remote_motor_mode(base: str) -> str:
    resp = requests.get(f"{base}/api/motors/status", timeout=5)
    resp.raise_for_status()
    return str(resp.json()["mode"])


def _wait_motors_disabled(base: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _remote_motor_mode(base) == "disabled":
            return True
        time.sleep(0.3)
    return False


@pytest.fixture
def head_raised_remote(daemon_running: str) -> Iterator[None]:
    """The `head_raised` fixture over REST: head up and verified before any sound.

    The robot idles with motors disabled and the head resting on the speaker,
    so the motors are enabled first; otherwise the goto is accepted and
    nothing moves.

    Teardown never cuts torque itself: doing so with the head up drops it.
    It runs the daemon's sleep move instead, which lowers the head under
    torque and releases it only at the rest pose.
    """
    base = f"http://{daemon_running}:8000"
    original_mode = _remote_motor_mode(base)

    requests.post(f"{base}/api/motors/set_mode/enabled", timeout=5).raise_for_status()
    requests.post(
        f"{base}/api/move/goto",
        json={
            "head_pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
            "antennas": list(INIT_ANTENNAS_JOINT_POSITIONS),
            "duration": 2.0,
        },
        timeout=5,
    ).raise_for_status()
    deadline = time.time() + 10
    while requests.get(f"{base}/api/move/running", timeout=5).json():
        if time.time() > deadline:
            pytest.fail("Head move did not finish within 10 s.")
        time.sleep(0.2)
    _assert_head_raised(_remote_head_pose(base)["z"])
    try:
        yield
    finally:
        if original_mode == "disabled":
            requests.post(f"{base}/api/move/play/goto_sleep", timeout=5)
            if not _wait_motors_disabled(base, 15):
                warnings.warn(
                    "Robot did not return to sleep; it is holding its pose "
                    "with torque on (safe, but not idle).",
                    stacklevel=2,
                )
