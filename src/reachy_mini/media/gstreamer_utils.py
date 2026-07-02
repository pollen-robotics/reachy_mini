"""Shared GStreamer helpers used by camera and audio backends."""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required but could not be imported. "
        "Please check the gstreamer installation."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstPbutils", "1.0")

from gi.repository import GLib, Gst, GstApp, GstPbutils  # noqa: E402


def handle_default_bus_message(
    logger: logging.Logger,
    msg: Gst.Message,
    pipeline: Gst.Pipeline,
) -> bool:
    """Handle GStreamer bus messages with sensible defaults.

    - ``EOS``: log a warning and return False (the bus watch is
      removed).
    - ``ERROR``: log the parsed error and return False.
    - ``WARNING``: log the parsed warning and keep the watch alive.
    - ``LATENCY``: call ``pipeline.recalculate_latency()`` and return
      True.
    - Anything else: return True (keep the watch alive).

    Callers can wrap this in their own handler to inject extra logic
    for a specific message type, then fall through to this helper for
    the common cases.
    """
    if msg.type == Gst.MessageType.EOS:
        logger.warning("End-of-stream")
        return False
    elif msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        logger.error(f"Error: {err} {debug}")
        return False
    elif msg.type == Gst.MessageType.WARNING:
        err, debug = msg.parse_warning()
        logger.warning(f"Warning: {err} {debug}")
    elif msg.type == Gst.MessageType.LATENCY:
        pipeline.recalculate_latency()
        logger.debug("Recalculate latency")
    return True


def get_sample(appsink: GstApp.AppSink, logger: logging.Logger) -> Optional[bytes]:
    """Pull a sample from a GStreamer AppSink with a 20 ms timeout.

    Args:
        appsink: The GStreamer AppSink element to pull from.
        logger: Logger for warnings.

    Returns:
        Raw bytes of the buffer, or ``None`` if no sample was available.

    """
    sample = appsink.try_pull_sample(20_000_000)
    if sample is None:
        return None
    data = None
    if isinstance(sample, Gst.Sample):
        buf = sample.get_buffer()
        if buf is None:
            logger.warning("Buffer is None")
        data = buf.extract_dup(0, buf.get_size())
    return data


def encode_bgr_to_jpeg(
    frame: npt.NDArray[np.uint8], logger: logging.Logger
) -> Optional[bytes]:
    """Encode a BGR frame as JPEG bytes, or ``None`` if encoding fails."""
    Gst.init([])  # idempotent
    height, width = frame.shape[:2]
    try:
        pipeline = Gst.parse_launch(
            "appsrc name=src format=time is-live=false "
            "! videoconvert ! jpegenc ! appsink name=sink sync=false"
        )
    except GLib.Error as e:
        logger.warning(f"Failed to build JPEG encoder pipeline: {e}")
        return None

    appsrc = pipeline.get_by_name("src")
    appsink = pipeline.get_by_name("sink")
    appsrc.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={width},height={height},framerate=1/1"
        ),
    )
    pipeline.set_state(Gst.State.PLAYING)
    try:
        appsrc.emit("push-buffer", Gst.Buffer.new_wrapped(frame.tobytes()))
        appsrc.emit("end-of-stream")
        return get_sample(appsink, logger)
    finally:
        pipeline.set_state(Gst.State.NULL)


def is_valid_audio_file(path: str) -> bool:
    """Return whether the file at *path* is a decodable audio file.

    Uses GStreamer's ``pbutils`` discoverer — the same machinery used to play
    sounds — so anything that passes here is guaranteed to be playable. The
    probe inspects the media's structure rather than trusting the extension,
    which rejects mislabelled or non-audio payloads.

    Fails closed: returns ``False`` if the probe errors out for any reason.

    Args:
        path: Filesystem path to the file to validate.

    Returns:
        ``True`` if the file is a decodable media file with at least one audio
        stream, ``False`` otherwise.

    """
    Gst.init([])  # idempotent
    try:
        discoverer = GstPbutils.Discoverer.new(5 * Gst.SECOND)
        info = discoverer.discover_uri(Gst.filename_to_uri(path))
    except Exception as e:
        logging.warning(f"Could not validate audio file {path}: {e}")
        return False

    return (
        info.get_result() == GstPbutils.DiscovererResult.OK
        and len(info.get_audio_streams()) > 0
    )


def audio_duration_seconds(path: str) -> float:
    """Return the media duration in seconds via GStreamer's discoverer.

    Works for any container the daemon can play (WAV, OGG, MP3, FLAC, ...).

    Args:
        path: Filesystem path to the media file.

    Returns:
        The duration in seconds.

    """
    Gst.init([])  # idempotent
    info = GstPbutils.Discoverer.new(5 * Gst.SECOND).discover_uri(
        Gst.filename_to_uri(path)
    )
    return float(info.get_duration()) / 1_000_000_000
