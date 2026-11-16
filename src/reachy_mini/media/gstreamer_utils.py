"""Shared GStreamer helpers used by camera and audio backends."""

import logging
import threading
from collections.abc import Callable
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
gi.require_version("GstPbutils", "1.0")

from gi.repository import GLib, Gst, GstApp, GstPbutils  # noqa: E402


class SoundPlaybin:
    """The single ``play_sound`` playbin, freed on EOS, error or replacement.

    An idle playbin left in PLAYING keeps the audio device open forever.
    Thread-safe: ``play``/``stop`` run on caller threads, the bus watch on
    the GLib main loop thread.
    """

    def __init__(
        self, logger: logging.Logger, on_end: Optional[Callable[[], None]] = None
    ) -> None:
        """Create an empty slot.

        Args:
            logger: Logger for playbin errors.
            on_end: Called on the GLib thread when a sound ends on its own.

        """
        self._logger = logger
        self._on_end = on_end
        self._lock = threading.Lock()
        self._playbin: Optional[Gst.Element] = None

    @property
    def current(self) -> Optional[Gst.Element]:
        """The playbin currently playing, if any."""
        return self._playbin

    def play(self, playbin: Gst.Element) -> None:
        """Free the previous playbin, then start *playbin*."""
        # add_watch, not a signal watch: returning False frees the closure.
        # A connected handler holding the playbin would leak it.
        playbin.get_bus().add_watch(GLib.PRIORITY_DEFAULT, self._on_message, playbin)
        with self._lock:
            old, self._playbin = self._playbin, playbin
        if old is not None:
            self._free(old)
        playbin.set_state(Gst.State.PLAYING)

    def stop(self) -> None:
        """Stop and free the current playbin. No-op if none."""
        with self._lock:
            old, self._playbin = self._playbin, None
        if old is not None:
            self._free(old)

    @staticmethod
    def _free(playbin: Gst.Element) -> None:
        # Returns False if the watch already ended, harmless.
        playbin.get_bus().remove_watch()
        playbin.set_state(Gst.State.NULL)

    def _on_message(
        self, _bus: Gst.Bus, msg: Gst.Message, playbin: Gst.Element
    ) -> bool:
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            self._logger.error(f"play_sound playbin error: {err} ({dbg})")
        elif msg.type != Gst.MessageType.EOS:
            return True
        with self._lock:
            mine = self._playbin is playbin
            if mine:
                self._playbin = None
        if mine:  # else play/stop already freed it
            playbin.set_state(Gst.State.NULL)
            if self._on_end is not None:
                self._on_end()
        return False


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


def get_sample(
    appsink: GstApp.AppSink,
    logger: logging.Logger,
    timeout_ns: int = 20_000_000,
) -> Optional[bytes]:
    """Pull a sample from a GStreamer AppSink with a bounded timeout.

    Args:
        appsink: The GStreamer AppSink element to pull from.
        logger: Logger for warnings.
        timeout_ns: How long to wait for a sample, in nanoseconds. Defaults to
            20 ms, which suits video-rate pulling where a missed frame is fine.
            One-shot encoders (e.g. ``read_jpeg``) need a larger value so the
            pipeline has time to preroll on a loaded machine.

    Returns:
        Raw bytes of the buffer, or ``None`` if no sample was available.

    """
    sample = appsink.try_pull_sample(timeout_ns)
    if sample is None:
        return None
    data = None
    if isinstance(sample, Gst.Sample):
        buf = sample.get_buffer()
        if buf is None:
            logger.warning("Buffer is None")
        data = buf.extract_dup(0, buf.get_size())
    return data


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
