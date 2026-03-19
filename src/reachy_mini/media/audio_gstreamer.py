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
    """Audio implementation using GStreamer."""

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
            id_audio_card = self._get_audio_device("Source")

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
        self._appsrc.set_property("is-live", True)
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
            id_audio_card = self._get_audio_device("Sink")

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
            id_audio_card = self._get_audio_device("Sink")
            audiosink = Gst.ElementFactory.make("wasapi2sink")
            audiosink.set_property("device", id_audio_card)
            self.logger.info(
                f"Using audio device {id_audio_card} for playback on Windows."
            )
        elif platform.system() == "Darwin":
            id_audio_card = self._get_audio_device("Sink")
            audiosink = Gst.ElementFactory.make("osxaudiosink")
            audiosink.set_property("unique-id", id_audio_card)
            self.logger.info(
                f"Using audio device {id_audio_card} for playback on macOS."
            )
        else:
            id_audio_card = self._get_audio_device("Sink")
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

    def __del__(self) -> None:
        """Ensure GStreamer resources are released."""
        self.cleanup()
        self._loop.quit()
        self._bus_record.remove_watch()
        self._bus_playback.remove_watch()

    def _get_audio_device(self, device_type: str = "Source") -> Optional[str]:
        """Use ``Gst.DeviceMonitor`` to find the Reachy Mini Audio card.

        Returns the platform-specific device identifier, or ``None``.
        """
        monitor = Gst.DeviceMonitor()
        monitor.add_filter(f"Audio/{device_type}")
        monitor.start()

        snd_card_name = "Reachy Mini Audio"
        try:
            devices = monitor.get_devices()
            for device in devices:
                name = device.get_display_name()
                device_props = device.get_properties()

                if snd_card_name in name:
                    # Skip monitor/loopback sources (e.g. "Monitor of Reachy Mini Audio")
                    if device_props and device_props.has_field("device.class"):
                        if device_props.get_string("device.class") == "monitor":
                            continue
                    if device_props and device_props.has_field("node.name"):
                        node_name = device_props.get_string("node.name")
                        self.logger.debug(
                            f"Found audio input device with node name {node_name}"
                        )
                        return str(node_name)
                    elif (
                        platform.system() == "Windows"
                        and device_props.has_field("device.api")
                        and device_props.get_string("device.api") == "wasapi2"
                    ):
                        if device_type == "Source" and device_props.get_value(
                            "wasapi2.device.loopback"
                        ):
                            continue  # skip loopback devices for source
                        device_id = device_props.get_string("device.id")
                        self.logger.debug(
                            f"Found audio input device {name} for Windows"
                        )
                        return str(device_id)
                    elif platform.system() == "Darwin":
                        device_id = device_props.get_string("unique-id")
                        self.logger.debug(f"Found audio input device {name} for macOS")
                        return str(device_id)
                    elif platform.system() == "Linux":
                        # Linux PulseAudio / ALSA fallback
                        # Construct PulseAudio device name from udev.id
                        udev_id = (
                            device_props.get_string("udev.id")
                            if device_props.has_field("udev.id")
                            else None
                        )
                        profile = (
                            device_props.get_string("device.profile.name")
                            if device_props.has_field("device.profile.name")
                            else None
                        )
                        if udev_id and profile:
                            prefix = (
                                "alsa_output" if device_type == "Sink" else "alsa_input"
                            )
                            pa_device = f"{prefix}.{udev_id}.{profile}"
                            self.logger.debug(
                                f"Found audio {device_type} device {name} via PulseAudio: {pa_device}"
                            )
                            return pa_device
                        elif device_props.has_field("device.string"):
                            device_id = device_props.get_string("device.string")
                            self.logger.debug(
                                f"Found audio {device_type} device {name} via ALSA: {device_id}"
                            )
                            return str(device_id)

            self.logger.warning(f"No Reachy Mini Audio {device_type} card found.")
        except Exception as e:
            self.logger.error(f"Error while getting audio input device: {e}")
        finally:
            monitor.stop()
        return None
