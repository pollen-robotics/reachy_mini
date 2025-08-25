import logging
import os
from threading import Thread
from typing import Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp


class GstPlayer:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self.pipeline = Gst.Pipeline.new("audio_player")

        # Optional device name from env (substring match)
        audio_out = os.getenv("AUDIO_OUT")

        # Create elements
        self.appsrc = Gst.ElementFactory.make("appsrc", None)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", True)
        caps = Gst.Caps.from_string(
            "audio/x-raw,format=S16LE,channels=1,rate=24000,layout=interleaved"
        )
        self.appsrc.set_property("caps", caps)
        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")

        # Try to pin specific output device; fallback to autoaudiosink
        audiosink = _create_device_element(
            direction="sink", name_substr=audio_out
        ) or Gst.ElementFactory.make("autoaudiosink")

        self.pipeline.add(self.appsrc)
        self.pipeline.add(queue)
        self.pipeline.add(audioconvert)
        self.pipeline.add(audioresample)
        self.pipeline.add(audiosink)

        self.appsrc.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(audiosink)

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error(f"Error: {err} {debug}")
            return False

        return True

    def _handle_bus_calls(self) -> None:
        self._logger.debug("starting bus message loop")
        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop)
        self._loop.run()  # type: ignore[no-untyped-call]
        bus.remove_watch()
        self._logger.debug("bus message loop stopped")

    def play(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()

    def push_sample(self, data: bytes):
        buf = Gst.Buffer.new_wrapped(data)
        self.appsrc.push_buffer(buf)

    def stop(self):
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)


class GstRecorder:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None

        self.pipeline = Gst.Pipeline.new("audio_recorder")

        audio_in = os.getenv("AUDIO_IN")

        # Create elements: try specific mic; fallback to default
        autoaudiosrc = _create_device_element(
            direction="source", name_substr=audio_in
        ) or Gst.ElementFactory.make("autoaudiosrc", None)

        queue = Gst.ElementFactory.make("queue", None)
        audioconvert = Gst.ElementFactory.make("audioconvert", None)
        audioresample = Gst.ElementFactory.make("audioresample", None)
        self.appsink = Gst.ElementFactory.make("appsink", None)

        if not all([autoaudiosrc, queue, audioconvert, audioresample, self.appsink]):
            raise RuntimeError("Failed to create GStreamer elements")

        # Force mono/S16LE at 24000; resample handles device SR (e.g., 16000 → 24000)
        caps = Gst.Caps.from_string("audio/x-raw,channels=1,rate=24000,format=S16LE")
        self.appsink.set_property("caps", caps)

        # Build pipeline
        self.pipeline.add(autoaudiosrc)
        self.pipeline.add(queue)
        self.pipeline.add(audioconvert)
        self.pipeline.add(audioresample)
        self.pipeline.add(self.appsink)

        autoaudiosrc.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(self.appsink)

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error(f"Error: {err} {debug}")
            return False

        return True

    def _handle_bus_calls(self) -> None:
        self._logger.debug("starting bus message loop")
        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop)
        self._loop.run()  # type: ignore[no-untyped-call]
        bus.remove_watch()
        self._logger.debug("bus message loop stopped")

    def record(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()

    def get_sample(self):
        sample = self.appsink.pull_sample()
        data = None
        if isinstance(sample, Gst.Sample):
            buf = sample.get_buffer()
            if buf is None:
                self._logger.warning("Buffer is None")

            data = buf.extract_dup(0, buf.get_size())
        return data

    def stop(self):
        self._loop.quit()
        self.pipeline.set_state(Gst.State.NULL)


def _create_device_element(
    direction: str, name_substr: Optional[str]
) -> Optional[Gst.Element]:
    """
    direction: 'source' or 'sink'
    name_substr: case-insensitive substring matching device display name/description.
    """
    if not name_substr:
        return None
    monitor = Gst.DeviceMonitor.new()
    klass = "Audio/Source" if direction == "source" else "Audio/Sink"
    monitor.add_filter(klass, None)
    monitor.start()
    try:
        for dev in monitor.get_devices() or []:
            disp = dev.get_display_name() or ""
            props = dev.get_properties()
            desc = (
                props.get_string("device.description")
                if props and props.has_field("device.description")
                else ""
            )
            if (
                name_substr.lower() in disp.lower()
                or name_substr.lower() in desc.lower()
            ):
                elem = dev.create_element(None)
                logging.getLogger(__name__).info(
                    "Using %s device: %s", direction, disp or desc
                )
                return elem
    finally:
        monitor.stop()
    logging.getLogger(__name__).warning(
        "Requested %s '%s' not found; using auto*", direction, name_substr
    )
    return None
