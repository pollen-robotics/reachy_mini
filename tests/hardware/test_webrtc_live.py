"""WebRTC media path against a real robot, driven from a machine on the LAN.

The robot's daemon serves camera and microphone over WebRTC (``webrtcsink`` +
its signalling server on port 8443) and accepts audio back to its speaker.
These tests connect exactly like a remote client (the JS apps, a telepresence
session) and gate on four things:

- **video is live**: frames keep arriving at a usable rate and actually change
  (a frozen pipeline delivering one stuck image fails),
- **microphone streams**: audio samples arrive at the expected rate,
- **the full duplex loop works**: a sweep pushed over WebRTC comes out of the
  robot's speaker, crosses the room, and returns through the mic over WebRTC —
  one test covering both network directions plus the speaker and mic,
- **the robot's pipeline latency is at its baseline**: GStreamer's own
  latency report for the robot's media pipeline, read over REST. It is
  configuration-derived and robot-side, so it does not move with the client
  or the network.

Network-dependent figures (the acoustic round trip, jitter, packet loss) are
out of scope: they vary with the client and the link, not the robot.

Requirements:

- a Reachy Mini Wireless on the LAN (``REACHY_TEST_HOST``, default
  ``reachy-mini.local``); the daemon is started automatically if stopped,
- the gst-plugins-rs webrtc plugin on *this* machine (``webrtcsrc``), which on
  a host that exports ``GST_PLUGIN_PATH`` from a shell profile means setting it
  explicitly for a non-interactive run,
- a reasonably quiet room for the duplex test.

Run from a repo checkout on the client machine::

    uv run pytest tests/hardware -v -m wireless

``REACHY_TEST_ARTIFACTS=<dir>`` saves the duplex capture (``webrtc_audio.wav``)
and its metrics (``webrtc_duplex.json``).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import gi
import numpy as np
import numpy.typing as npt
import pytest
import requests

gi.require_version("Gst", "1.0")
gi.require_version("GstRtp", "1.0")
# GstWebRTC is imported for its types: without it the remote description comes
# back as an opaque GBoxed with no ``.sdp``.
gi.require_version("GstWebRTC", "1.0")
from gi.repository import Gst, GstRtp, GstWebRTC  # noqa: E402,F401

from reachy_mini.media.audio_utils import save_audio_to_wav  # noqa: E402
from reachy_mini.media.media_manager import MediaBackend, MediaManager  # noqa: E402
from tests.audio_helpers import (  # noqa: E402
    correlation_peak,
    log_sweep,
    loudest_block_rms,
    noise_floor,
)

RATE = 16000
SWEEP_S = 2.0
SWEEP = log_sweep(SWEEP_S, 100.0, 7800.0, RATE, 0.7)

# Gates. The audio-rate tolerance sits under the nominal 16 kHz; the duplex
# numbers are provisional until measured over enough runs (the test prints
# every metric — tighten from real data).
#
# RECEIVED_FPS_MIN gates the rate the robot actually delivers: distinct H.264
# frame timestamps entering the client's RTP jitter buffer, straight off the
# network. Counting there (not after decode, not even at the depayloader, which
# a slow decoder backpressures) makes it independent of the client: a
# Raspberry Pi 4 decodes ~14 of the 1080p frames per second it receives, yet
# both it and a laptop measure 30 here. Nominal is 30 fps; 24 leaves room for
# jitter over a short window.
RECEIVED_FPS_MIN = 24.0
# Decoded frames only have to prove the stream is valid, not keep up.
DECODED_FRAMES_MIN = 3
AUDIO_RATE_TOL = 0.15
# Duplex "the sound came back" gates. Over WebRTC the sweep crosses Opus
# twice plus the room, which flattens its correlation with the reference:
# healthy runs measured xcorr 0.076-0.236, against 0.024-0.040 for noise with
# no sweep in it. 0.05 sits between the two. BURST_MIN is the level check
# (the sweep actually played); healthy runs measured 42-254.
XCORR_MIN = 0.05
BURST_MIN = 5.0

# GStreamer's report of the *robot's* sender pipeline latency, read over REST
# from /api/media/latency. Configuration-derived (element latencies + jitter
# buffers) and robot-side, so unlike the acoustic round trip it does not move
# with the client machine or the network. That is what makes it worth
# baselining.
#
# Measured on a Reachy Mini Wireless, 20 samples: 38.0-38.9 ms with no
# consumer attached, 36.0-37.6 ms with one. The ~2.5 ms step is why the
# baseline is centred between the two states rather than pinned to either;
# the tolerance only has to absorb that step, since a real graph change (an
# added buffer or element) moves this by tens of ms.
ROBOT_PIPELINE_LATENCY_MS = 37.5
ROBOT_PIPELINE_LATENCY_TOL_MS = 6.0

BLOCK_SAMPLES = 800  # 50 ms at 16 kHz
FIRST_MEDIA_TIMEOUT_S = 15.0
VIDEO_WINDOW_S = 3.0
AUDIO_WINDOW_S = 2.0


@pytest.fixture
def webrtc_media(daemon_running: str) -> "MediaManager":
    """A WebRTC client connected to the robot, closed after the test."""
    media = MediaManager(backend=MediaBackend.WEBRTC, signalling_host=daemon_running)
    try:
        yield media
    finally:
        media.close()


def _wait_first(get, what: str):
    """Poll ``get()`` until it returns something, or fail."""
    deadline = time.monotonic() + FIRST_MEDIA_TIMEOUT_S
    while True:
        value = get()
        if value is not None:
            return value
        if time.monotonic() > deadline:
            pytest.fail(f"No {what} within {FIRST_MEDIA_TIMEOUT_S}s of connecting.")
        time.sleep(0.02)


def _robot_pipeline_latency_ms(host: str) -> float:
    """The robot's own sender-pipeline latency, over the daemon REST API."""
    resp = requests.get(f"http://{host}:8000/api/media/latency", timeout=10)
    if resp.status_code == 503:
        pytest.fail(f"Robot media pipeline has no latency to report: {resp.text}")
    resp.raise_for_status()
    return float(resp.json()["latency_ms"])


@pytest.mark.wireless
def test_robot_pipeline_latency_matches_baseline(daemon_running: str) -> None:
    """The robot's media pipeline latency is at its baseline.

    A robot-side property: no client connection and no media flow needed, so
    nothing here depends on this machine or the network.
    """
    latency_ms = _robot_pipeline_latency_ms(daemon_running)
    drift = latency_ms - ROBOT_PIPELINE_LATENCY_MS
    print(f"\nrobot pipeline latency: {latency_ms:.3f}ms ({drift:+.3f}ms vs baseline)")

    assert abs(drift) <= ROBOT_PIPELINE_LATENCY_TOL_MS, (
        f"robot pipeline latency {latency_ms:.1f}ms drifted {drift:+.1f}ms from "
        f"the {ROBOT_PIPELINE_LATENCY_MS}ms baseline (tolerance "
        f"±{ROBOT_PIPELINE_LATENCY_TOL_MS}): a jitter buffer or an element in "
        "the robot's media graph changed."
    )


def _h264_payload_type(media: MediaManager) -> int:
    """The negotiated RTP payload type for H.264, read from the remote SDP."""
    webrtcbin = media.camera._webrtcbin
    description = webrtcbin.get_property("remote-description")
    if description is None:
        pytest.fail("WebRTC session has no remote description yet.")
    match = re.search(r"a=rtpmap:(\d+) H264/90000", description.sdp.as_text())
    if match is None:
        pytest.fail("Remote SDP offers no H.264 video.")
    return int(match.group(1))


def _jitterbuffer_sink_pads(media: MediaManager) -> list[Gst.Pad]:
    """Input pads of every RTP jitter buffer in the client pipeline.

    Reaches into the client's pipeline: the SDK exposes decoded frames only.
    """
    pads = []
    it = media.camera._pipeline_record.iterate_recurse()
    while True:
        result, element = it.next()
        if result != Gst.IteratorResult.OK:
            return pads
        factory = element.get_factory()
        if factory is not None and factory.get_name() == "rtpjitterbuffer":
            pads.append(element.get_static_pad("sink"))


@pytest.mark.wireless
def test_video_stream_is_live(webrtc_media: MediaManager) -> None:
    """The robot delivers ~30 fps, and the stream decodes into valid frames.

    The rate is counted on frames arriving from the network, before any
    decoding, so it measures the robot and the link rather than how fast this
    client decodes. Decoding only
    has to show a few distinct, correctly sized frames (a frozen pipeline
    serving one stuck image fails).
    """
    media = webrtc_media
    first = _wait_first(media.get_frame, "video frame")
    assert first.ndim == 3 and first.shape[2] == 3, f"bad frame shape {first.shape}"
    resolution = media.camera.resolution
    assert first.shape[:2] == (resolution[1], resolution[0]), (
        f"frame {first.shape[:2]} != advertised resolution {resolution}"
    )

    # One frame spans several RTP packets sharing a timestamp, and FEC packets
    # use their own payload type, so count distinct H.264 timestamps.
    video_pt = _h264_payload_type(media)
    timestamps: set[int] = set()

    def count(_pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        ok, rtp = GstRtp.RTPBuffer.map(info.get_buffer(), Gst.MapFlags.READ)
        if ok:
            if rtp.get_payload_type() == video_pt:
                timestamps.add(rtp.get_timestamp())
            rtp.unmap()
        return Gst.PadProbeReturn.OK

    probes = [
        (pad, pad.add_probe(Gst.PadProbeType.BUFFER, count))
        for pad in _jitterbuffer_sink_pads(media)
    ]

    # Distinct decoded frames, compared on a sparse pixel grid: enough to tell
    # a new frame from a videorate duplicate, and cheap on a small client.
    distinct = 0
    previous = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < VIDEO_WINDOW_S:
        frame = media.get_frame()
        if frame is None:
            continue
        current = frame[::40, ::40, 0].copy()
        if previous is None or not np.array_equal(current, previous):
            distinct += 1
        previous = current
    for pad, probe_id in probes:
        pad.remove_probe(probe_id)

    received_fps = len(timestamps) / VIDEO_WINDOW_S
    print(
        f"\nvideo: {received_fps:.1f} fps received from the robot, "
        f"{distinct / VIDEO_WINDOW_S:.1f} fps decoded on this client"
    )
    assert received_fps >= RECEIVED_FPS_MIN, (
        f"robot delivers {received_fps:.1f} fps < {RECEIVED_FPS_MIN} (nominal 30)"
    )
    assert distinct >= DECODED_FRAMES_MIN, (
        f"only {distinct} distinct decoded frames — the stream does not decode, "
        "or the video is frozen"
    )


@pytest.mark.wireless
def test_audio_stream_from_microphone(webrtc_media: MediaManager) -> None:
    """Mic samples arrive continuously at the advertised rate and channels."""
    media = webrtc_media
    _wait_first(media.get_audio_sample, "audio sample")

    chunks = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < AUDIO_WINDOW_S:
        sample = media.get_audio_sample()
        if sample is not None:
            chunks.append(sample)
        else:
            time.sleep(0.002)

    audio = np.concatenate(chunks, axis=0)
    rate = media.get_input_audio_samplerate()
    expected = AUDIO_WINDOW_S * rate
    print(
        f"\naudio: {audio.shape[0]} samples in {AUDIO_WINDOW_S}s "
        f"({audio.shape[0] / AUDIO_WINDOW_S:.0f} Hz effective), "
        f"{audio.shape[1]} channels"
    )

    assert audio.shape[1] == media.get_input_channels()
    assert audio.shape[0] == pytest.approx(expected, rel=AUDIO_RATE_TOL), (
        f"sample flow {audio.shape[0]} != ~{expected:.0f} — stream is stalling"
    )
    assert float(np.abs(audio).max()) > 0.0, "audio stream is all zeros"


@pytest.mark.wireless
def test_full_duplex_acoustic_loopback(
    aec_disabled_remote: object,
    head_raised_remote: None,
    pinned_volume_remote: None,
    webrtc_media: MediaManager,
) -> None:
    """A sweep pushed over WebRTC returns through the robot's mic.

    Covers, in one loop: client->robot audio (network + decode + speaker),
    the acoustic path, and robot->client audio (mic + encode + network).
    Passes when the sweep comes back loud and recognisable; how long the
    trip takes depends on the network and is not checked.
    """
    media = webrtc_media
    _wait_first(media.get_audio_sample, "audio sample")

    chunks: list[npt.NDArray[np.float32]] = []

    def drain() -> None:
        while (sample := media.get_audio_sample()) is not None:
            chunks.append(sample)

    # Pre-push window: the noise floor the burst level is measured against.
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.5:
        drain()
        time.sleep(0.002)
    samples_before_push = sum(c.shape[0] for c in chunks)

    media.start_playing()
    media.push_audio_sample(SWEEP.astype(np.float32))
    t0 = time.monotonic()
    while time.monotonic() - t0 < SWEEP_S + 3.0:
        drain()
        time.sleep(0.002)
    media.stop_playing()

    audio = np.concatenate(chunks, axis=0).astype(np.float64)
    track = audio[:, 0]
    noise = track[:samples_before_push]

    xcorr, _ = correlation_peak(track, SWEEP, RATE)
    burst = loudest_block_rms(track[samples_before_push:], BLOCK_SAMPLES) / (
        noise_floor(noise) + 1e-12
    )
    print(
        f"\nduplex: xcorr={xcorr:.3f} burst={burst:.1f} "
        f"(capture {len(track)} samples, {samples_before_push} pre-push)"
    )

    artifacts = os.environ.get("REACHY_TEST_ARTIFACTS")
    if artifacts:
        out = Path(artifacts)
        out.mkdir(parents=True, exist_ok=True)
        save_audio_to_wav(audio.astype(np.float32), RATE, str(out / "webrtc_audio.wav"))
        (out / "webrtc_duplex.json").write_text(
            json.dumps(
                {
                    "xcorr": xcorr,
                    "burst": burst,
                    "samples_before_push": samples_before_push,
                    "rate_hz": RATE,
                },
                indent=1,
            )
        )
        print(f"artifacts written to {out}")

    failures = []
    if xcorr <= XCORR_MIN:
        failures.append(
            f"the sweep never came back (xcorr={xcorr:.3f} <= {XCORR_MIN}): "
            "client->speaker or mic->client audio is broken, or echo "
            "cancellation is still active."
        )
    if burst <= BURST_MIN:
        failures.append(
            f"nothing loud reached the mic (burst={burst:.1f} <= {BURST_MIN}): "
            "the pushed audio never played on the speaker."
        )
    assert not failures, "\n".join(failures)
