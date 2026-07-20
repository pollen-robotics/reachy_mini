"""End-to-end WebRTC loopback: a fake robot producer <-> GstWebRTCClient.

Simulates the robot with a plain ``gst-launch-1.0 webrtcsink`` pipeline (raw
video/audio + its embedded signalling server) — no need to boot the real
daemon. The real code under test is :class:`GstWebRTCClient` (``webrtcsrc``):
it discovers the producer, negotiates over the signalling server, and receives
media. Exercises the live paths (open/pad-added/receive chains/audio send
setup) that the mocked unit tests can't.

Requires the gst-plugins-rs webrtc plugin (``libgstrswebrtc.so``); skipped
where absent, so it runs only where the plugin is installed.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import numpy as np
import pytest

from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs
from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient
from reachy_mini.media.webrtc_utils import find_producer_peer_id_by_name

pytestmark = pytest.mark.webrtc

_PORT = 8443
_NAME = "reachymini"


@pytest.fixture
def fake_robot():
    """A `gst-launch webrtcsink` producer + embedded signalling server.

    Yields the producer's peer id once it has registered on the server.
    """
    if shutil.which("gst-launch-1.0") is None:
        pytest.skip("gst-launch-1.0 not available")

    # A subprocess, not Gst.parse_launch in-process: keep the producer's
    # webrtcsink + signalling server + libnice/DTLS stack isolated from the
    # client-under-test's stack. That mirrors the real robot<->client process
    # split, gives clean port/teardown, and contains crashes (a plugin segfault
    # fails the fixture instead of killing pytest). gst-launch also runs the bus
    # + main loop for us, so it's less code than driving the pipeline by hand.
    proc = subprocess.Popen(
        [
            "gst-launch-1.0",
            "-q",
            "webrtcsink",
            "name=ws",
            f"meta=meta,name={_NAME}",
            "run-signalling-server=true",
            f"signalling-server-port={_PORT}",
            "videotestsrc",
            "is-live=true",
            "!",
            "video/x-raw,width=640,height=480",
            "!",
            "ws.",
            "audiotestsrc",
            "is-live=true",
            "!",
            "audio/x-raw,rate=48000,channels=2",
            "!",
            "ws.",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    peer_id = None
    for _ in range(60):  # up to ~30s for the producer to register
        if proc.poll() is not None:
            break
        try:
            peer_id = find_producer_peer_id_by_name("127.0.0.1", _PORT, _NAME)
            if peer_id:
                break
        except (KeyError, OSError, ConnectionError):
            pass
        time.sleep(0.5)

    if not peer_id:
        proc.terminate()
        out = proc.communicate(timeout=5)[0]
        pytest.fail(
            f"fake robot producer did not register:\n{out.decode(errors='ignore')[:800]}"
        )

    yield peer_id

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_client_receives_video_and_audio_over_webrtc(fake_robot) -> None:
    """GstWebRTCClient negotiates with the producer and receives real media.

    Pulls a video frame (BGR ndarray) and confirms the audio receive path
    yields non-silent samples — the producer sends audiotestsrc (a tone) over
    Opus, which the client decodes to its audio appsink.
    """
    client = GstWebRTCClient(
        peer_id=fake_robot,
        signaling_host="127.0.0.1",
        signaling_port=_PORT,
        camera_specs=ReachyMiniLiteCamSpecs(),
    )
    frame = None
    audio_chunks: list[np.ndarray] = []
    try:
        client.open()
        deadline = time.time() + 25  # ICE/DTLS + first media
        while time.time() < deadline and (frame is None or len(audio_chunks) < 5):
            f = client.read()
            if f is not None and frame is None:
                frame = f
            sample = client.get_audio_sample()
            if sample is not None:
                audio_chunks.append(sample)
            time.sleep(0.1)
    finally:
        client.close()
        client._loop.quit()

    assert frame is not None, "no video frame received over the WebRTC loopback"
    assert frame.ndim == 3 and frame.shape[2] == 3

    assert audio_chunks, "no audio received over the WebRTC loopback"
    audio = np.concatenate(audio_chunks, axis=0).astype(np.float64)
    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms > 1e-3, f"received audio is silent (rms={rms})"
