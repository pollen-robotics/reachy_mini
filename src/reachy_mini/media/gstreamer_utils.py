"""Shared GStreamer helpers used by camera and audio backends."""

import logging
from typing import Optional

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required but could not be imported. "
        "Please check the gstreamer installation."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import Gst, GstApp  # noqa: E402


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
