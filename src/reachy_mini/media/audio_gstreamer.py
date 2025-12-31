"""GStreamer audio backend.

This module provides an implementation of the AudioBase class using GStreamer.
By default the module directly returns audio stream by ReStreamer
"""

import os
import sys
from threading import Thread
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_utils import (
    get_respeaker_card_number,
    has_reachymini_asoundrc,
)
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required for GStreamerAudio but could not be imported. \
                      Please install the GStreamer backend: pip install .[gstreamer]."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")


from gi.repository import GLib, Gst, GstApp  # noqa: E402

from .audio_base import AudioBase  # noqa: E402


class GStreamerAudio(AudioBase):
    """Audio implementation using GStreamer."""

    DEFAULT_LATENCY_US = 30000
    DEFAULT_BUFFER_US = 200000

    def __init__(self, log_level: str = "INFO", alsa_pcm_type: str = "plughw") -> None:
        """Initialize the GStreamer audio."""
        super().__init__(log_level=log_level)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self._is_linux = sys.platform.startswith("linux")

        self._id_audio_card = get_respeaker_card_number()
        self._alsa_pcm_type = alsa_pcm_type
        self._pipeline_record = Gst.Pipeline.new("audio_recorder")
        self._appsink_audio: Optional[GstApp] = None
        self._init_pipeline_record(self._pipeline_record)
        self._bus_record = self._pipeline_record.get_bus()
        self._bus_record.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        self._pipeline_playback = Gst.Pipeline.new("audio_player")
        self._appsrc: Optional[GstApp] = None
        self._init_pipeline_playback(self._pipeline_playback)
        self._bus_playback = self._pipeline_playback.get_bus()
        self._bus_playback.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

    def _init_pipeline_record(self, pipeline: Gst.Pipeline) -> None:
        self._appsink_audio = Gst.ElementFactory.make("appsink")
        
        # 1. Define the target caps (What we WANT: 16k, F32LE)
        caps_string = (
            f"audio/x-raw,rate={self.SAMPLE_RATE},"
            f"channels={self.CHANNELS},"
            f"format=F32LE,layout=interleaved"
        )
        caps = Gst.Caps.from_string(caps_string)
        
        # Configure AppSink
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)
        self._appsink_audio.set_property("max-buffers", 200)

        if self._is_linux and self._id_audio_card != -1:
            self.logger.info(f"Using alsasrc device {self._alsa_pcm_type}:{self._id_audio_card},0")
            audiosrc = Gst.ElementFactory.make("alsasrc")
            audiosrc.set_property("device", f"{self._alsa_pcm_type}:{self._id_audio_card},0")
            
            audiosrc.set_property("latency-time", self.DEFAULT_LATENCY_US)
            audiosrc.set_property("buffer-time", self.DEFAULT_BUFFER_US)
        else:
            self.logger.info("Using autoaudiosrc (System Default)")
            audiosrc = Gst.ElementFactory.make("autoaudiosrc")

        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")
        
        elements = [audiosrc, queue, audioconvert, audioresample, self._appsink_audio]
        if not all(elements):
            raise RuntimeError("Failed to create required GStreamer elements")

        for elem in elements:
            pipeline.add(elem)

        if not audiosrc.link(queue):
            raise RuntimeError("Failed to link audiosrc -> queue")
        if not queue.link(audioconvert):
            raise RuntimeError("Failed to link queue -> audioconvert")
        if not audioconvert.link(audioresample):
            raise RuntimeError("Failed to link audioconvert -> audioresample")
        if not audioresample.link(self._appsink_audio):
            raise RuntimeError("Failed to link audioresample -> appsink")

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        super().__del__()
        self._loop.quit()
        self._bus_record.remove_watch()
        self._bus_playback.remove_watch()

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

        queue = Gst.ElementFactory.make("queue")
        audiosink: Optional[Gst.Element] = None
        if self._id_audio_card == -1:
            audiosink = Gst.ElementFactory.make("autoaudiosink")  # use default speaker
        else:
            audiosink = Gst.ElementFactory.make("alsasink")
            audiosink.set_property("device", f"{self._alsa_pcm_type}:{self._id_audio_card},0")

        pipeline.add(queue)
        pipeline.add(audiosink)
        pipeline.add(self._appsrc)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)

        self._appsrc.link(queue)
        queue.link(audioconvert)
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
        """Open the audio card using GStreamer."""
        self._pipeline_record.set_state(Gst.State.PLAYING)

    def _get_sample(self, appsink: GstApp.AppSink) -> Optional[npt.NDArray[np.float32]]:
        data = None
        sample = appsink.try_pull_sample(20_000_000)

        if isinstance(sample, Gst.Sample):
            buf = sample.get_buffer()
            if buf is not None:
                success, mapinfo = buf.map(Gst.MapFlags.READ)
                if success:
                    try:
                        data = np.frombuffer(mapinfo.data, dtype=np.float32).copy()
                    finally:
                        buf.unmap(mapinfo)
                else:
                    self.logger.error("Failed to map buffer")
            else:
                self.logger.warning("Buffer is None")

        return data

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read a sample from the audio card."""
        if self._appsink_audio is None:
            return None
        data = self._get_sample(self._appsink_audio)
        if data is None:
            return None
            
        try:
            return data.reshape(-1, self.CHANNELS)
        except ValueError as e:
            self.logger.error(f"Shape mismatch! Buffer size doesn't match channels. Error: {e}")
            return None

    def get_input_audio_samplerate(self) -> int:
        """Get the input samplerate of the audio device."""
        return self.SAMPLE_RATE

    def get_output_audio_samplerate(self) -> int:
        """Get the output samplerate of the audio device."""
        return self.SAMPLE_RATE

    def get_input_channels(self) -> int:
        """Get the number of input channels of the audio device."""
        return self.CHANNELS

    def get_output_channels(self) -> int:
        """Get the number of output channels of the audio device."""
        return self.CHANNELS

    def stop_recording(self) -> None:
        """Release the audio resource."""
        self._pipeline_record.set_state(Gst.State.NULL)

    def start_playing(self) -> None:
        """Open the audio output using GStreamer."""
        self._pipeline_playback.set_state(Gst.State.PLAYING)

    def stop_playing(self) -> None:
        """Stop playing audio and release resources."""
        self._pipeline_playback.set_state(Gst.State.NULL)

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output device."""
        if self._appsrc is not None:
            buf = Gst.Buffer.new_wrapped(data.tobytes())
            self._appsrc.push_buffer(buf)
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

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
