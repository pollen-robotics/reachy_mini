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

import gi
import pytest

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init([])
if (
    Gst.ElementFactory.find("webrtcsrc") is None
    or Gst.ElementFactory.find("webrtcsink") is None
):
    pytest.skip(
        "gst-plugins-rs webrtc (libgstrswebrtc.so) not available",
        allow_module_level=True,
    )

from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs  # noqa: E402
from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient  # noqa: E402
from reachy_mini.media.webrtc_utils import (  # noqa: E402
    find_producer_peer_id_by_name,
)

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


def test_client_receives_video_over_webrtc(fake_robot) -> None:
    """GstWebRTCClient negotiates with the producer and pulls a real frame."""
    client = GstWebRTCClient(
        peer_id=fake_robot,
        signaling_host="127.0.0.1",
        signaling_port=_PORT,
        camera_specs=ReachyMiniLiteCamSpecs(),
    )
    frame = None
    try:
        client.open()
        deadline = time.time() + 25  # ICE/DTLS + first frame
        while time.time() < deadline:
            f = client.read()
            if f is not None:
                frame = f
                break
            time.sleep(0.2)
    finally:
        client.close()
        client._loop.quit()

    assert frame is not None, "no video frame received over the WebRTC loopback"
    assert frame.ndim == 3 and frame.shape[2] == 3
