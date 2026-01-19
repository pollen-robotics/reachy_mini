"""GStreamer audio backend.

This module provides an implementation of the AudioBase class using GStreamer.
It offers advanced audio processing capabilities including microphone input,
speaker output, and integration with the ReSpeaker microphone array for
Direction of Arrival (DoA) estimation.

The GStreamer audio backend supports:
- High-quality audio capture and playback
- ReSpeaker microphone array integration
- Direction of Arrival (DoA) estimation
- Advanced audio processing pipelines
- Multiple audio formats and sample rates

Note:
    This class is typically used internally by the MediaManager when the GSTREAMER
    backend is selected. Direct usage is possible but usually not necessary.

Example usage via MediaManager:
    >>> from reachy_mini.media.media_manager import MediaManager, MediaBackend
    >>>
    >>> # Create media manager with GStreamer backend
    >>> media = MediaManager(backend=MediaBackend.GSTREAMER, log_level="INFO")
    >>>
    >>> # Start audio recording
    >>> media.start_recording()
    >>>
    >>> # Get audio samples
    >>> samples = media.get_audio_sample()
    >>> if samples is not None:
    ...     print(f"Captured {len(samples)} audio samples")
    >>>
    >>> # Get Direction of Arrival
    >>> doa = media.get_DoA()
    >>> if doa is not None:
    ...     angle, speech_detected = doa
    ...     print(f"Sound direction: {angle} radians, speech detected: {speech_detected}")
    >>>
    >>> # Clean up
    >>> media.stop_recording()
    >>> media.close()

"""

import os
import time
from threading import Thread
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_utils import (
    get_audio_device_node_name,
    has_reachymini_asoundrc,
)
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required for GStreamerCamera but could not be imported. \
                      Please install the GStreamer backend: pip install .[gstreamer]."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")


from gi.repository import GLib, Gst, GstApp  # noqa: E402

from .audio_base import AudioBase  # noqa: E402


class GStreamerAudio(AudioBase):
    """Audio implementation using GStreamer."""

    def __init__(self, input_device: str | None = None, output_device: str | None = None, log_level: str = "INFO") -> None:
        """Initialize the GStreamer audio."""
        super().__init__(log_level=log_level)
        Gst.init(None)

        # Initialize all instance variables first (for safe __del__)
        self._loop: Optional[GLib.MainLoop] = None
        self._thread_bus_calls: Optional[Thread] = None
        self._pipeline_record: Optional[Gst.Pipeline] = None
        self._pipeline_playback: Optional[Gst.Pipeline] = None
        self._bus_record: Optional[Gst.Bus] = None
        self._bus_playback: Optional[Gst.Bus] = None
        self._appsink_audio: Optional[GstApp] = None
        self._appsrc: Optional[GstApp] = None

        self._loop = GLib.MainLoop()
        loop = self._loop  # capture for lambda to satisfy mypy
        self._thread_bus_calls = Thread(target=lambda: loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self._audio_input_node_name = get_audio_device_node_name(device_class="Audio/Source")
        self._audio_output_node_name = get_audio_device_node_name(device_class="Audio/Sink")

        self._pipeline_record = Gst.Pipeline.new("audio_recorder")
        self._init_pipeline_record(self._pipeline_record, input_device)
        self._bus_record = self._pipeline_record.get_bus()
        self._bus_record.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        self._pipeline_playback = Gst.Pipeline.new("audio_player")
        self._init_pipeline_playback(self._pipeline_playback, output_device)
        self._bus_playback = self._pipeline_playback.get_bus()
        self._bus_playback.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

    def _init_pipeline_record(self, pipeline: Gst.Pipeline, input_device: str | None = None) -> None:
        self._appsink_audio = Gst.ElementFactory.make("appsink")
        caps = Gst.Caps.from_string(
            f"audio/x-raw,rate={self.SAMPLE_RATE},channels={self.CHANNELS},format=F32LE,layout=interleaved"
        )
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)  # avoid overflow
        self._appsink_audio.set_property("max-buffers", 200)

        audiosrc: Optional[Gst.Element] = None
        input_node_name: Optional[str] = None
        if input_device is not None:
            input_node_name = get_audio_device_node_name(device_names=[input_device], device_class="Audio/Source")

        if input_node_name is not None:
            self.logger.info(f"Using pipewire source node {input_node_name}")
            audiosrc = Gst.ElementFactory.make("pulsesrc")
            audiosrc.set_property("device", f"{input_node_name}")
        elif self._audio_input_node_name is None:
            audiosrc = Gst.ElementFactory.make("autoaudiosrc")  # use default mic
        elif has_reachymini_asoundrc():
            # reachy mini wireless has a preconfigured asoundrc
            audiosrc = Gst.ElementFactory.make("alsasrc")
            audiosrc.set_property("device", "reachymini_audio_src")
        else:
            audiosrc = Gst.ElementFactory.make("pulsesrc")
            audiosrc.set_property("device", f"{self._audio_input_node_name}")

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

    def _cleanup_pipelines(self) -> None:
        """Clean up pipelines and bus watches (but keep the main loop running)."""
        # Stop pipelines first
        if self._pipeline_record is not None:
            self._pipeline_record.set_state(Gst.State.NULL)
        if self._pipeline_playback is not None:
            self._pipeline_playback.set_state(Gst.State.NULL)
        # Remove bus watches
        if self._bus_record is not None:
            self._bus_record.remove_watch()
            self._bus_record = None
        if self._bus_playback is not None:
            self._bus_playback.remove_watch()
            self._bus_playback = None
        # Clear pipeline references
        self._pipeline_record = None
        self._pipeline_playback = None
        self._appsink_audio = None
        self._appsrc = None

    def close(self) -> None:
        """Explicitly close and release all GStreamer resources."""
        self._cleanup_pipelines()
        # Quit the main loop
        if self._loop is not None and self._loop.is_running():
            self._loop.quit()

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        super().__del__()
        self._cleanup_pipelines()
        # Quit the main loop last
        if self._loop is not None and self._loop.is_running():
            self._loop.quit()

    def set_max_output_buffers(self, max_buffers: int) -> None:
        """Set the maximum number of output buffers to queue in the player.

        Args:
            max_buffers (int): Maximum number of buffers to queue.

        """
        if self._appsrc is not None:
            self._appsrc.set_property("max-buffers", max_buffers)
            self._appsrc.set_property("leaky-type", 2)  # drop old buffers
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def _init_pipeline_playback(self, pipeline: Gst.Pipeline, output_device: str | None = None) -> None:
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
        output_node_name: Optional[str] = None
        if output_device is not None:
            output_node_name = get_audio_device_node_name(device_names=[output_device], device_class="Audio/Sink")

        if output_node_name is not None:
            self.logger.info(f"Using pipewire sink node {output_node_name}")
            audiosink = Gst.ElementFactory.make("pulsesink")
            audiosink.set_property("device", f"{output_node_name}")
        elif self._audio_output_node_name is None:
            audiosink = Gst.ElementFactory.make("autoaudiosink")  # use default speaker
        elif has_reachymini_asoundrc():
            # reachy mini wireless has a preconfigured asoundrc
            audiosink = Gst.ElementFactory.make("alsasink")
            audiosink.set_property("device", "reachymini_audio_sink")
        else:
            audiosink = Gst.ElementFactory.make("pulsesink")
            audiosink.set_property("device", f"{self._audio_output_node_name}")

        pipeline.add(audiosink)
        pipeline.add(self._appsrc)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)

        self._appsrc.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(audiosink)

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

    def start_recording(self) -> None:
        """Open the audio card using GStreamer.

        See AudioBase.start_recording() for complete documentation.
        """
        if self._pipeline_record is not None:
            self._pipeline_record.set_state(Gst.State.PLAYING)
            time.sleep(1)

    def _get_sample(self, appsink: GstApp.AppSink) -> Optional[bytes]:
        sample = appsink.try_pull_sample(20_000_000)
        if sample is None:
            return None
        data = None
        if isinstance(sample, Gst.Sample):
            buf = sample.get_buffer()
            if buf is None:
                self.logger.warning("Buffer is None")

            data = buf.extract_dup(0, buf.get_size())
        return data

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read a sample from the audio card. Returns the sample or None if error.

        See AudioBase.get_audio_sample() for complete documentation.

        Returns:
            Optional[npt.NDArray[np.float32]]: The captured sample in raw format, or None if error.

        """
        sample = self._get_sample(self._appsink_audio)
        if sample is None:
            return None
        return np.frombuffer(sample, dtype=np.float32).reshape(-1, 2)

    def get_input_audio_samplerate(self) -> int:
        """Get the input samplerate of the audio device.

        See AudioBase.get_input_audio_samplerate() for complete documentation.
        """
        return self.SAMPLE_RATE

    def get_output_audio_samplerate(self) -> int:
        """Get the output samplerate of the audio device.

        See AudioBase.get_output_audio_samplerate() for complete documentation.
        """
        return self.SAMPLE_RATE

    def get_input_channels(self) -> int:
        """Get the number of input channels of the audio device.

        See AudioBase.get_input_channels() for complete documentation.
        """
        return self.CHANNELS

    def get_output_channels(self) -> int:
        """Get the number of output channels of the audio device.

        See AudioBase.get_output_channels() for complete documentation.
        """
        return self.CHANNELS

    def stop_recording(self) -> None:
        """Release the camera resource.

        See AudioBase.stop_recording() for complete documentation.
        """
        if self._pipeline_record is not None:
            self._pipeline_record.set_state(Gst.State.NULL)

    def start_playing(self) -> None:
        """Open the audio output using GStreamer.

        See AudioBase.start_playing() for complete documentation.
        """
        if self._pipeline_playback is not None:
            self._pipeline_playback.set_state(Gst.State.PLAYING)
            time.sleep(1)

    def stop_playing(self) -> None:
        """Stop playing audio and release resources.

        See AudioBase.stop_playing() for complete documentation.
        """
        if self._pipeline_playback is not None:
            self._pipeline_playback.set_state(Gst.State.NULL)

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output device.

        See AudioBase.push_audio_sample() for complete documentation.
        """
        if self._appsrc is not None:
            buf = Gst.Buffer.new_wrapped(data.tobytes())
            self._appsrc.push_buffer(buf)
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        See AudioBase.play_sound() for complete documentation.

        Todo: for now this function is mean to be used on the wireless version.

        Args:
            sound_file (str): Path to the sound file to play.

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

        playbin = Gst.ElementFactory.make("playbin", "player")
        if not playbin:
            self.logger.error("Failed to create playbin element")
            return
        playbin.set_property("uri", f"file://{file_path}")
        if audiosink is not None:
            playbin.set_property("audio-sink", audiosink)

        playbin.set_state(Gst.State.PLAYING)

    def clear_player(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        if self._appsrc is not None and self._pipeline_playback is not None:
            # Push a silent sample to unstick pulsesink before flushing
            num_samples = 1024
            silence = np.zeros((num_samples, self.CHANNELS), dtype=np.float32)
            self.push_audio_sample(silence)
            time.sleep(num_samples / self.SAMPLE_RATE)

            self._pipeline_playback.set_state(Gst.State.PAUSED)
            self._appsrc.send_event(Gst.Event.new_flush_start())
            self._appsrc.send_event(Gst.Event.new_flush_stop(reset_time=True))
            self._pipeline_playback.set_state(Gst.State.PLAYING)
            self.logger.info("Cleared player queue")
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )
