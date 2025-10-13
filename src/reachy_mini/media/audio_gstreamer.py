"""GStreamer camera backend.

This module provides an implementation of the CameraBase class using GStreamer.
By default the module directly returns JPEG images as output by the camera.
"""

from threading import Thread
from typing import Optional

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

from .audio_base import AudioBackend, AudioBase  # noqa: E402


class GStreamerAudio(AudioBase):
    """Audio implementation using GStreamer."""

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the GStreamer audio."""
        super().__init__(backend=AudioBackend.GSTREAMER, log_level=log_level)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        self._samplerate = 24000

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
        caps = Gst.Caps.from_string(
            f"audio/x-raw,channels=1,rate={self._samplerate},format=S16LE"
        )
        self._appsink_audio.set_property("caps", caps)
        self._appsink_audio.set_property("drop", True)  # avoid overflow
        self._appsink_audio.set_property("max-buffers", 200)

        autoaudiosrc = Gst.ElementFactory.make("autoaudiosrc")  # use default mic
        # caps_respeaker = Gst.Caps.from_string(
        #    "audio/x-raw, layout=interleaved, format=S16LE, rate=16000, channels=2"
        # )
        # autoaudiosrc.set_property("filter-caps", caps_respeaker)
        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")

        if not all(
            [autoaudiosrc, queue, audioconvert, audioresample, self._appsink_audio]
        ):
            raise RuntimeError("Failed to create GStreamer elements")

        pipeline.add(autoaudiosrc)
        pipeline.add(queue)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(self._appsink_audio)

        autoaudiosrc.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(self._appsink_audio)

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        self._loop.quit()
        self._bus_record.remove_watch()
        self._bus_playback.remove_watch()

    def _init_pipeline_playback(self, pipeline: Gst.Pipeline) -> None:
        self._appsrc = Gst.ElementFactory.make("appsrc")
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("is-live", True)
        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,channels=1,rate={self._samplerate},layout=interleaved"
        )
        self._appsrc.set_property("caps", caps)

        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")

        queue = Gst.ElementFactory.make("queue")
        audiosink = Gst.ElementFactory.make("autoaudiosink")  # use default speaker

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

    def _get_sample(self, appsink: Gst.AppSink) -> Optional[bytes]:
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

    def get_audio_sample(self) -> Optional[bytes]:
        """Read a sample from the audio card. Returns the sample or None if error.

        Returns:
            Optional[bytes]: The captured sample in raw format, or None if error.

        """
        return self._get_sample(self._appsink_audio)

    def get_audio_samplerate(self) -> int:
        """Return the samplerate of the audio device."""
        return self._samplerate

    def stop_recording(self) -> None:
        """Release the camera resource."""
        self._pipeline_record.set_state(Gst.State.NULL)

    def start_playing(self) -> None:
        """Open the audio output using GStreamer."""
        self._pipeline_playback.set_state(Gst.State.PLAYING)

    def stop_playing(self) -> None:
        """Stop playing audio and release resources."""
        self._pipeline_playback.set_state(Gst.State.NULL)

    def push_audio_sample(self, data: bytes) -> None:
        """Push audio data to the output device."""
        if self._appsrc is not None:
            buf = Gst.Buffer.new_wrapped(data)
            self._appsrc.push_buffer(buf)
        else:
            self.logger.warning(
                "AppSrc is not initialized. Call start_playing() first."
            )

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play.

        """
        self.logger.warning("play_sound is not implemented for GStreamerAudio.")
