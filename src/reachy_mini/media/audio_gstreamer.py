"""GStreamer audio backend.

Handles microphone input, speaker output, and sound-file playback using
GStreamer pipelines.  Also provides Direction of Arrival (DoA) estimation
via the ReSpeaker microphone array (see ``AudioDoA``).

Recording pipeline::

    platform_source → queue → audioconvert → audioresample → appsink(F32LE)

Playback pipeline::

    appsrc(F32LE) → audioconvert → audioresample → platform_sink

Platform audio sources / sinks are discovered at runtime:

* **Linux (PipeWire / PulseAudio)**: ``pulsesrc`` / ``pulsesink``
* **Linux (ALSA, Reachy Mini Wireless)**: ``alsasrc`` / ``alsasink``
  with the preconfigured ``reachymini_audio_src`` / ``reachymini_audio_sink``
  devices from ``~/.asoundrc``.
* **Windows**: ``wasapi2src`` / ``wasapi2sink``
* **macOS**: ``osxaudiosrc`` / ``osxaudiosink``
* **Fallback**: ``autoaudiosrc`` / ``autoaudiosink``

The "Reachy Mini Audio" card is located by name via ``Gst.DeviceMonitor``.
If no matching card is found the platform default is used instead.

Note:
    This class is typically used internally by ``MediaManager`` when the
    ``LOCAL`` backend is selected.  Direct usage is possible but usually
    not necessary.

Example usage via MediaManager::

    from reachy_mini.media.media_manager import MediaManager, MediaBackend

    media = MediaManager(backend=MediaBackend.LOCAL)
    media.start_recording()

    samples = media.get_audio_sample()
    if samples is not None:
        print(f"Captured {len(samples)} audio samples")

    doa = media.get_DoA()
    if doa is not None:
        angle, speech_detected = doa
        print(f"Sound direction: {angle} rad, speech detected: {speech_detected}")

    media.stop_recording()
    media.close()

"""

import os
import platform
from threading import Thread
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.audio_utils import has_reachymini_asoundrc
from reachy_mini.media.device_detection import get_audio_device
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required for GStreamerAudio but could not be imported. "
        "Please check the gstreamer installation."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst  # noqa: E402


class GStreamerAudio(AudioBase):
    """Audio implementation using GStreamer.

    Extends ``AudioBase`` with two GStreamer-specific helpers:

    - ``clear_output_buffer()``: flush queued playback data without stopping
      the pipeline (no-op by default; useful before refilling the buffer).
    - ``clear_player()``: flush the playback appsrc immediately via GStreamer
      flush events, dropping any queued audio.

    """

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize recording and playback pipelines.

        Args:
            log_level: Logging level for audio operations.
                Options: ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``,
                ``'CRITICAL'``.

        """
        super().__init__(log_level=log_level)

        Gst.init([])
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self._pipeline_record = Gst.Pipeline.new("audio_recorder")
        self._init_pipeline_record(self._pipeline_record)
        self._bus_record = self._pipeline_record.get_bus()
        self._bus_record.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        self._playbin: Optional[Gst.Element] = None
        self._pipeline_playback = Gst.Pipeline.new("audio_player")
        self._init_pipeline_playback(self._pipeline_playback)
        self._bus_playback = self._pipeline_playback.get_bus()
        self._bus_playback.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

    def _init_pipeline_record(self, pipeline: Gst.Pipeline) -> None:
        self._appsink_audio = Gst.ElementFactory.make("appsink")
        caps = Gst.Caps.from_string(
            f"audio/x-raw,rate={self.SAMPLE_RATE},channels={self.CHANNELS},format=F32LE,layout=interleaved"
        )
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)  # avoid overflow
        self._appsink_audio.set_property("max-buffers", 200)

        audiosrc: Optional[Gst.Element] = None

        if has_reachymini_asoundrc():
            # Wireless CM4: use the preconfigured .asoundrc ALSA devices
            # which route through the XMOS AEC loopback properly.
            audiosrc = Gst.ElementFactory.make("alsasrc")
            audiosrc.set_property("device", "reachymini_audio_src")
            self.logger.info("Using .asoundrc audio source: reachymini_audio_src")
        else:
            id_audio_card = get_audio_device("Source")

            if id_audio_card is None:
                self.logger.warning(
                    "No specific audio card found, using default audio source."
                )
                audiosrc = Gst.ElementFactory.make("autoaudiosrc")  # use default mic
            elif platform.system() == "Windows":
                audiosrc = Gst.ElementFactory.make("wasapi2src")
                audiosrc.set_property("device", id_audio_card)
            elif platform.system() == "Darwin":
                audiosrc = Gst.ElementFactory.make("osxaudiosrc")
                audiosrc.set_property("unique-id", id_audio_card)
            else:
                audiosrc = Gst.ElementFactory.make("pulsesrc")
                audiosrc.set_property("device", f"{id_audio_card}")

        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")

        if not all([audiosrc, queue, audioconvert, audioresample, self._appsink_audio]):
            raise RuntimeError("Failed to create GStreamer elements")

        pipeline.add(audiosrc)
        pipeline.add(queue)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(self._appsink_audio)

        audiosrc.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(self._appsink_audio)

    def _init_pipeline_playback(self, pipeline: Gst.Pipeline) -> None:
        self._appsrc = Gst.ElementFactory.make("appsrc")
        self._appsrc.set_property("format", Gst.Format.TIME)
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels={self.CHANNELS},rate={self.SAMPLE_RATE},layout=interleaved"
        )
        self._appsrc.set_property("caps", caps)

        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")

        audiosink: Optional[Gst.Element] = None

        if has_reachymini_asoundrc():
            # Wireless CM4: use the preconfigured .asoundrc ALSA devices
            # which route through the XMOS AEC loopback properly.
            audiosink = Gst.ElementFactory.make("alsasink")
            audiosink.set_property("device", "reachymini_audio_sink")
            self.logger.info("Using .asoundrc audio sink: reachymini_audio_sink")
        else:
            id_audio_card = get_audio_device("Sink")

            if id_audio_card is None:
                self.logger.warning(
                    "No specific audio card found, using default audio sink."
                )
                audiosink = Gst.ElementFactory.make(
                    "autoaudiosink"
                )  # use default speaker
            elif platform.system() == "Windows":
                audiosink = Gst.ElementFactory.make("wasapi2sink")
                audiosink.set_property("device", id_audio_card)
            elif platform.system() == "Darwin":
                audiosink = Gst.ElementFactory.make("osxaudiosink")
                audiosink.set_property("unique-id", id_audio_card)
            else:
                audiosink = Gst.ElementFactory.make("pulsesink")
                audiosink.set_property("device", f"{id_audio_card}")

        queue = Gst.ElementFactory.make("queue")

        pipeline.add(audiosink)
        pipeline.add(self._appsrc)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(queue)

        self._appsrc.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(queue)
        queue.link(audiosink)

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self.logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self.logger.error(f"Error: {err} {debug}")
            return False

        return True

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self._pipeline_playback.query(query)
        self.logger.info(f"Audio pipeline latency {query.parse_latency()}")

    def start_recording(self) -> None:
        """Start capturing audio from the microphone."""
        self._pipeline_record.set_state(Gst.State.PLAYING)

    def stop_recording(self) -> None:
        """Stop the recording pipeline."""
        self._pipeline_record.set_state(Gst.State.NULL)

    def start_playing(self) -> None:
        """Start the playback pipeline so ``push_audio_sample`` can feed data."""
        self._pipeline_playback.set_state(Gst.State.PLAYING)
        GLib.timeout_add_seconds(5, self._dump_latency)

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the speaker.

        Args:
            data: Audio samples as a float32 array.  Shape should be
                ``(num_samples, 2)`` for stereo or ``(num_samples,)`` for
                mono (the caller is responsible for channel adaptation).

        """
        if self._appsrc is not None:
            buf = Gst.Buffer.new_wrapped(data.tobytes())
            self._appsrc.push_buffer(buf)
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def stop_playing(self) -> None:
        """Stop the playback pipeline."""
        self._pipeline_playback.set_state(Gst.State.NULL)
        if self._playbin is not None:
            self._playbin.set_state(Gst.State.NULL)
            self._playbin = None

    def clear_output_buffer(self) -> None:
        """Flush queued playback data so it is not played.

        A low ``set_max_output_buffers`` value may make this unnecessary
        for most use-cases.

        """
        pass  # subclasses or future implementations can override

    def clear_player(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        if self._appsrc is not None:
            self._pipeline_playback.set_state(Gst.State.PAUSED)
            self._appsrc.send_event(Gst.Event.new_flush_start())
            self._appsrc.send_event(Gst.Event.new_flush_stop(reset_time=True))
            self._pipeline_playback.set_state(Gst.State.PLAYING)
            self.logger.info("Cleared player queue")
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file through the Reachy Mini Audio card.

        The file is played via a GStreamer ``playbin`` routed to the same
        audio sink used by the push-based playback pipeline.

        Args:
            sound_file: Absolute path **or** filename relative to the
                built-in assets directory.

        Raises:
            FileNotFoundError: If the file cannot be found.

        """
        if not os.path.exists(sound_file):
            file_path = f"{ASSETS_ROOT_PATH}/{sound_file}"
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Sound file {sound_file} not found in assets directory or given path."
                )
        else:
            file_path = sound_file

        audiosink: Optional[Gst.Element] = None

        if has_reachymini_asoundrc():
            # reachy mini wireless has a preconfigured asoundrc
            audiosink = Gst.ElementFactory.make("alsasink")
            audiosink.set_property("device", "reachymini_audio_sink")
            self.logger.info("Using audio device reachymini_audio_sink for playback.")
        elif platform.system() == "Windows":
            id_audio_card = get_audio_device("Sink")
            audiosink = Gst.ElementFactory.make("wasapi2sink")
            audiosink.set_property("device", id_audio_card)
            self.logger.info(
                f"Using audio device {id_audio_card} for playback on Windows."
            )
        elif platform.system() == "Darwin":
            id_audio_card = get_audio_device("Sink")
            audiosink = Gst.ElementFactory.make("osxaudiosink")
            audiosink.set_property("unique-id", id_audio_card)
            self.logger.info(
                f"Using audio device {id_audio_card} for playback on macOS."
            )
        else:
            id_audio_card = get_audio_device("Sink")
            audiosink = Gst.ElementFactory.make("pulsesink")
            audiosink.set_property("device", f"{id_audio_card}")
            self.logger.info(f"Using audio device {id_audio_card} for playback.")

        if self._playbin is not None:
            self._playbin.set_state(Gst.State.NULL)

        playbin = Gst.ElementFactory.make("playbin", "player")
        if not playbin:
            self.logger.error("Failed to create playbin element")
            return

        # Fix for Windows: use file:/// and forward slashes
        if os.name == "nt":
            uri_path = file_path.replace("\\", "/")
            if not uri_path.startswith("/") and ":" in uri_path:
                # Ensure three slashes after file: for absolute paths (file:///C:/...)
                uri = f"file:///{uri_path}"
            else:
                uri = f"file://{uri_path}"
        else:
            uri = f"file://{file_path}"
        playbin.set_property("uri", uri)
        if audiosink is not None:
            playbin.set_property("audio-sink", audiosink)

        self._playbin = playbin
        playbin.set_state(Gst.State.PLAYING)

    def upload_sound(self, sound_file: str) -> str:
        """No-op for the local backend — the file is already accessible.

        Returns:
            The unchanged *sound_file* path.

        """
        return sound_file

    def list_sounds(self) -> list[str]:
        """No-op for the local backend.

        Returns:
            An empty list.

        """
        return []

    def delete_sound(self, filename: str) -> bool:
        """No-op for the local backend.

        Returns:
            Always ``False``.

        """
        return False

    def get_DoA(self) -> tuple[float, bool] | None:
        """Get the Direction of Arrival (DoA) from the ReSpeaker.

        Returns:
            A tuple ``(angle_radians, speech_detected)`` or ``None``
            if the device is unavailable.

        """
        return self._doa.get_DoA()

    def cleanup(self) -> None:
        """Release all resources (pipelines, USB devices)."""
        self._doa.close()

    def __del__(self) -> None:
        """Ensure GStreamer resources are released."""
        self.cleanup()
        self._loop.quit()
        self._bus_record.remove_watch()
        self._bus_playback.remove_watch()
