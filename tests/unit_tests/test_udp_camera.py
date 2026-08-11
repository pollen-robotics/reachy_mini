"""UDP RTP camera sender test — the MuJoCo/sim video path, no camera device.

``GStreamerUDPCamera`` pays raw frames into RTP and sends them over UDP (this is
what the sim feeds the daemon on port 5005). We loop it back to a localhost
receiver built with the same caps the daemon's ``_build_sim_source`` uses, and
assert a frame arrives intact — pure userspace, no ``/dev/video``.
"""

import socket
import time

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp  # noqa: E402, F401

from reachy_mini.media.gstreamer_udp_camera import GStreamerUDPCamera  # noqa: E402


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_udp_camera_frame_round_trip() -> None:
    """Frames sent by GStreamerUDPCamera arrive intact on a UDP receiver."""
    import pytest

    Gst.init([])
    for name in ("udpsrc", "udpsink", "rtpvrawpay", "rtpvrawdepay", "videoconvert"):
        if Gst.ElementFactory.find(name) is None:
            pytest.skip(f"GStreamer element {name!r} not available")

    w, h = 320, 240
    port = _free_udp_port()
    # Same RTP caps the daemon's _build_sim_source declares for the sim feed.
    caps = (
        "application/x-rtp,media=(string)video,clock-rate=(int)90000,"
        "encoding-name=(string)RAW,sampling=(string)RGB,depth=(string)8,"
        f"width=(string){w},height=(string){h},payload=(int)96"
    )
    receiver = Gst.parse_launch(
        f'udpsrc port={port} caps="{caps}" ! rtpvrawdepay ! videoconvert '
        "! video/x-raw,format=RGB ! appsink name=sink sync=false max-buffers=4 drop=true"
    )
    sink = receiver.get_by_name("sink")
    receiver.set_state(Gst.State.PLAYING)

    cam = GStreamerUDPCamera(dest_ip="127.0.0.1", dest_port=port, width=w, height=h)
    cam.start()

    frame = np.full((h, w, 3), 123, dtype=np.uint8)  # solid grey, RGB
    received = None
    t0 = time.time()
    try:
        while time.time() - t0 < 5.0:
            cam.send_frame(frame)
            sample = sink.try_pull_sample(100 * Gst.MSECOND)
            if sample is not None:
                buf = sample.get_buffer()
                ok, info = buf.map(Gst.MapFlags.READ)
                try:
                    received = np.frombuffer(info.data, dtype=np.uint8).copy()
                finally:
                    buf.unmap(info)
                break
    finally:
        cam.close()
        receiver.set_state(Gst.State.NULL)

    assert received is not None, "no frame received over the UDP loopback"
    assert received.size >= w * h * 3
    # raw RTP is lossless, so the solid grey survives the round-trip
    assert abs(int(received[: w * h * 3].mean()) - 123) <= 2
