"""Unit tests for privacy mode.

No hardware needed: the audio chip is replaced by a fake, and the media
filters are exercised on test sources in small real GStreamer pipelines.
"""

import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import gi
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_mini.daemon.app.dependencies import get_daemon  # noqa: E402
from reachy_mini.daemon.app.routers import media, privacy  # noqa: E402
from reachy_mini.daemon.privacy import PrivacyMode  # noqa: E402
from reachy_mini.io.protocol import SetPrivacyCmd  # noqa: E402
from reachy_mini.media import audio_control_utils  # noqa: E402
from reachy_mini.media.camera_constants import CameraResolution  # noqa: E402
from reachy_mini.media.media_server import GstMediaServer  # noqa: E402

Gst.init([])


# ------------------------------------------------------------ the controller


class FakeRobot:
    """Records what privacy mode does to the microphones, camera and speaker."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def mute_microphones(self, muted: bool) -> bool:
        self.events.append(("mic_muted", muted))
        return True

    def set_privacy(self, enabled: bool) -> None:
        self.events.append(("media_privacy", enabled))

    def play_sound(self, sound_file: str) -> None:
        self.events.append(("sound", sound_file))


def make_privacy(robot: FakeRobot, tmp_path: Path) -> PrivacyMode:
    return PrivacyMode(
        robot,
        state_path=tmp_path / "privacy.json",
        mute_microphones=robot.mute_microphones,
    )


def test_enabling_cuts_microphones_and_camera_before_the_sound(tmp_path: Path) -> None:
    robot = FakeRobot()
    mode = make_privacy(robot, tmp_path)
    robot.events.clear()

    assert mode.set("api", True) is True

    assert mode.enabled
    assert robot.events == [
        ("mic_muted", True),
        ("media_privacy", True),
        ("sound", "privacy_on.wav"),
    ]


def test_disabling_restores_microphones_and_camera(tmp_path: Path) -> None:
    robot = FakeRobot()
    mode = make_privacy(robot, tmp_path)
    mode.set("api", True)
    robot.events.clear()

    assert mode.set("api", False) is False

    assert robot.events == [
        ("mic_muted", False),
        ("media_privacy", False),
        ("sound", "privacy_off.wav"),
    ]


def test_privacy_stays_on_until_every_source_clears(tmp_path: Path) -> None:
    robot = FakeRobot()
    mode = make_privacy(robot, tmp_path)
    mode.set("hat", True)
    mode.set("api", True)
    robot.events.clear()

    # Software cannot switch off a hat that is still on the head.
    assert mode.set("api", False) is True
    assert robot.events == []
    assert mode.sources == ["hat"]

    assert mode.set("hat", False) is False
    assert ("mic_muted", False) in robot.events


def test_repeating_a_request_does_not_replay_the_sound(tmp_path: Path) -> None:
    robot = FakeRobot()
    mode = make_privacy(robot, tmp_path)
    mode.set("api", True)
    robot.events.clear()

    mode.set("api", True)

    assert robot.events == []


def test_an_api_request_survives_a_restart_silently(tmp_path: Path) -> None:
    make_privacy(FakeRobot(), tmp_path).set("api", True)

    robot = FakeRobot()
    restarted = make_privacy(robot, tmp_path)

    assert restarted.enabled
    assert robot.events == [("mic_muted", True), ("media_privacy", True)]


def test_a_hat_request_is_not_restored_after_a_restart(tmp_path: Path) -> None:
    # The reader reports the hat again within one poll if it is still there.
    make_privacy(FakeRobot(), tmp_path).set("hat", True)

    robot = FakeRobot()
    restarted = make_privacy(robot, tmp_path)

    assert not restarted.enabled
    # A crash while private must not leave the microphones muted for ever.
    assert robot.events == [("mic_muted", False), ("media_privacy", False)]


def test_privacy_works_without_a_media_server(tmp_path: Path) -> None:
    mutes: list[bool] = []
    mode = PrivacyMode(
        None,
        state_path=tmp_path / "privacy.json",
        mute_microphones=lambda muted: mutes.append(muted) or True,
    )

    assert mode.set("api", True) is True
    assert mutes == [False, True]


# ----------------------------------------------------------- the microphones


class FakeChip:
    """An audio chip that remembers its mute pin and output routing."""

    def __init__(self, pin_wired: bool = True) -> None:
        self.pin_wired = pin_wired
        self.pins = [0, 0, 0, 1, 0]  # X0D11, X0D30, X0D31, X0D33, X0D39
        self.outputs = {"AUDIO_MGR_OP_L": (8, 0), "AUDIO_MGR_OP_R": (8, 0)}
        self.closed = False

    def write(self, name: str, values: list[int]) -> None:
        if name == "GPO_WRITE_VALUE":
            if self.pin_wired and values[0] == 30:
                self.pins[1] = values[1]
        else:
            self.outputs[name] = tuple(values)

    def read_values(self, name: str) -> tuple[int, ...]:
        if name == "GPO_READ_VALUES":
            return tuple(self.pins)
        return self.outputs[name]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def chip(monkeypatch: pytest.MonkeyPatch) -> FakeChip:
    chip = FakeChip()
    monkeypatch.setattr(audio_control_utils, "init_respeaker_usb", lambda: chip)
    monkeypatch.setattr(audio_control_utils, "WRITE_SETTLE_SECONDS", 0)
    return chip


def test_muting_cuts_the_microphones_and_silences_the_output(chip: FakeChip) -> None:
    assert audio_control_utils.set_microphones_muted(True) is True

    assert chip.pins[1] == 1  # hardware mute circuit
    assert chip.outputs == {"AUDIO_MGR_OP_L": (0, 0), "AUDIO_MGR_OP_R": (0, 0)}
    assert chip.closed


def test_unmuting_restores_the_microphones(chip: FakeChip) -> None:
    audio_control_utils.set_microphones_muted(True)

    assert audio_control_utils.set_microphones_muted(False) is True

    assert chip.pins[1] == 0
    assert chip.outputs == {"AUDIO_MGR_OP_L": (8, 0), "AUDIO_MGR_OP_R": (8, 0)}


def test_unmuting_leaves_a_custom_output_routing_alone(chip: FakeChip) -> None:
    # The daemon unmutes at every start: it must only undo its own mute.
    chip.outputs["AUDIO_MGR_OP_L"] = (3, 0)

    assert audio_control_utils.set_microphones_muted(False) is True

    assert chip.outputs["AUDIO_MGR_OP_L"] == (3, 0)


def test_muting_reports_failure_when_the_chip_does_not_follow(chip: FakeChip) -> None:
    chip.pin_wired = False

    assert audio_control_utils.set_microphones_muted(True) is False


def test_muting_reports_failure_without_an_audio_board(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_control_utils, "init_respeaker_usb", lambda: None)

    assert audio_control_utils.set_microphones_muted(True) is False


# ---------------------------------------------------------- the media server


def _make_server() -> GstMediaServer:
    """Build a minimal ``GstMediaServer`` without booting its pipeline."""
    server = cast(GstMediaServer, object.__new__(GstMediaServer))
    server._logger = logging.getLogger("test_privacy")
    server._resolution = CameraResolution.R1280x720at30fps
    # Destructor (__del__ -> close()) touches these at GC time.
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    return server


def _pull(appsink: Gst.Element, count: int) -> bytes:
    """Pull ``count`` buffers and return the last one."""
    data = b""
    for _ in range(count):
        sample = appsink.emit("try-pull-sample", 5 * Gst.SECOND)
        assert sample is not None, "the pipeline produced no buffer"
        buffer = sample.get_buffer()
        data = buffer.extract_dup(0, buffer.get_size())
    return data


def test_privacy_blacks_out_camera_frames_and_restores_them() -> None:
    server = _make_server()
    pipeline = Gst.Pipeline.new("privacy_video_test")
    source = Gst.ElementFactory.make("videotestsrc")
    source.set_property("pattern", "white")
    caps = Gst.ElementFactory.make("capsfilter")
    caps.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw,format=I420,width=1280,height=720"),
    )
    appsink = Gst.ElementFactory.make("appsink")
    appsink.set_property("max-buffers", 1)
    for element in (source, caps, appsink):
        pipeline.add(element)
    source.link(caps)
    server._add_privacy_filter(pipeline, caps).link(appsink)

    def luma() -> np.ndarray:
        frame = np.frombuffer(_pull(appsink, 4), dtype=np.uint8)
        return frame[: 1280 * 720].reshape(720, 1280)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        assert luma().min() > 200  # the white test picture goes through

        server.set_privacy(True)
        private = luma()
        # Everything outside the centred label is black.
        assert private[:200].max() == 0 and private[-200:].max() == 0
        assert private[:, :200].max() == 0 and private[:, -200:].max() == 0
        if server._privacy_label is not None:
            assert private[260:460, 256:1024].max() > 200  # the label is drawn

        server.set_privacy(False)
        assert luma().min() > 200
    finally:
        pipeline.set_state(Gst.State.NULL)


def test_privacy_silences_the_streamed_microphone() -> None:
    server = _make_server()
    pipeline = Gst.Pipeline.new("privacy_audio_test")
    source = Gst.ElementFactory.make("audiotestsrc")
    caps = Gst.ElementFactory.make("capsfilter")
    caps.set_property("caps", Gst.Caps.from_string("audio/x-raw,format=S16LE"))
    appsink = Gst.ElementFactory.make("appsink")
    appsink.set_property("max-buffers", 1)
    for element in (source, caps, appsink):
        pipeline.add(element)
    source.link(caps)
    server._add_privacy_mute(pipeline, caps).link(appsink)

    def peak() -> int:
        return int(np.abs(np.frombuffer(_pull(appsink, 4), dtype=np.int16)).max())

    pipeline.set_state(Gst.State.PLAYING)
    try:
        assert peak() > 1000
        server.set_privacy(True)
        assert peak() == 0
        server.set_privacy(False)
        assert peak() > 1000
    finally:
        pipeline.set_state(Gst.State.NULL)


def test_a_rebuilt_pipeline_starts_private() -> None:
    # acquire_media() rebuilds the pipeline from scratch: it must not come
    # back with a live camera while privacy mode is on.
    server = _make_server()
    server.set_privacy(True)
    pipeline = Gst.Pipeline.new("privacy_rebuild_test")
    source = Gst.ElementFactory.make("videotestsrc")
    pipeline.add(source)

    server._add_privacy_filter(pipeline, source)

    assert server._privacy_black.get_property("brightness") == -1.0


# ------------------------------------------------------------- the REST API


@pytest.fixture
def api(tmp_path: Path) -> tuple[TestClient, FakeRobot]:
    robot = FakeRobot()
    daemon = MagicMock()
    daemon.privacy = make_privacy(robot, tmp_path)
    app = FastAPI()
    app.include_router(privacy.router)
    app.include_router(media.router)
    app.dependency_overrides[get_daemon] = lambda: daemon
    return TestClient(app), robot


def test_rest_api_switches_privacy_on_and_off(
    api: tuple[TestClient, FakeRobot],
) -> None:
    client, robot = api
    assert client.get("/privacy").json() == {"enabled": False, "sources": []}

    response = client.post("/privacy", json={"enabled": True})
    assert response.status_code == 200
    assert response.json() == {"enabled": True, "sources": ["api"]}
    assert ("mic_muted", True) in robot.events

    assert client.post("/privacy", json={"enabled": False}).json() == {
        "enabled": False,
        "sources": [],
    }


def test_media_cannot_be_released_while_private(
    api: tuple[TestClient, FakeRobot],
) -> None:
    # Releasing hands the camera and the sound card to the caller.
    client, _ = api
    client.post("/privacy", json={"enabled": True})

    assert client.post("/media/release").status_code == 409


# ------------------------------------------------- the data-channel command


def test_set_privacy_command_reaches_the_daemon(sim_backend: Any) -> None:
    requests: list[bool] = []
    sim_backend.set_privacy_callback(lambda enabled: requests.append(enabled) or True)
    responses: list[dict[str, Any]] = []

    sim_backend.process_command(
        SetPrivacyCmd(enabled=True), send_response=responses.append
    )

    assert requests == [True]
    assert responses == [{"status": "ok", "command": "set_privacy", "enabled": True}]
