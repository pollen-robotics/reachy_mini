"""WebRTC daemon.

Starts a gstreamer webrtc pipeline to stream video and audio.
"""

import logging
from threading import Thread
from typing import Optional, Tuple, cast

import gi

from reachy_mini.media.audio_utils import get_respeaker_card_number
from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
    ReachyMiniWirelessCamSpecs,
)

# from reachy_mini.media.camera_utils import get_video_device

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst  # noqa: E402


class GstWebRTC:
    """WebRTC pipeline using GStreamer."""

    def __init__(
        self,
        localhost: bool = False,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the GStreamer WebRTC pipeline."""
        self._localhost = localhost
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        cam_path, self.camera_specs = self.get_video_device()

        if self.camera_specs is None:
            raise RuntimeError("Camera specs not set")
        self._resolution = self.camera_specs.default_resolution
        self.resized_K = self.camera_specs.K

        if self._resolution is None:
            raise RuntimeError("Failed to get default camera resolution.")

        self._id_audio_card = get_respeaker_card_number()

        self._pipeline_sender = Gst.Pipeline.new("reachymini_webrtc_sender")
        self._bus_sender = self._pipeline_sender.get_bus()
        self._bus_sender.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        webrtcsink = self._configure_webrtc(self._pipeline_sender)

        self._configure_video(cam_path, self._pipeline_sender, webrtcsink)
        self._configure_audio(self._pipeline_sender, webrtcsink)

        self._pipeline_receiver = Gst.Pipeline.new("reachymini_webrtc_receiver")
        self._bus_receiver = self._pipeline_receiver.get_bus()
        self._bus_receiver.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )
        self._configure_receiver(self._pipeline_receiver)

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        self._loop.quit()
        self._bus_sender.remove_watch()
        self._bus_receiver.remove_watch()

    def _configure_webrtc(self, pipeline: Gst.Pipeline) -> Gst.Element:
        self._logger.debug("Configuring WebRTC")
        webrtcsink = Gst.ElementFactory.make("webrtcsink")
        if not webrtcsink:
            raise RuntimeError(
                "Failed to create webrtcsink element. Is the GStreamer webrtc rust plugin installed?"
            )

        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", "reachymini")
        webrtcsink.set_property("meta", meta_structure)
        webrtcsink.set_property("run-signalling-server", True)
        webrtcsink.set_property("congestion-control", 0 if self._localhost else 1)
        # webrtcsink.set_property("start-bitrate", 5_000_000)

        pipeline.add(webrtcsink)

        return webrtcsink

    def _configure_receiver(self, pipeline: Gst.Pipeline) -> None:
        udpsrc = Gst.ElementFactory.make("udpsrc")
        udpsrc.set_property("port", 5000)
        caps = Gst.Caps.from_string(
            "application/x-rtp,media=audio,encoding-name=OPUS,payload=96"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)
        rtpjitterbuffer = Gst.ElementFactory.make("rtpjitterbuffer")
        rtpjitterbuffer.set_property(
            "latency", 200
        )  # configure latency depending on network conditions
        rtpopusdepay = Gst.ElementFactory.make("rtpopusdepay")
        opusdec = Gst.ElementFactory.make("opusdec")
        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")
        alsasink = Gst.ElementFactory.make("alsasink")
        alsasink.set_property("device", f"hw:{self._id_audio_card},0")
        alsasink.set_property("sync", False)

        pipeline.add(udpsrc)
        pipeline.add(capsfilter)
        pipeline.add(rtpjitterbuffer)
        pipeline.add(rtpopusdepay)
        pipeline.add(opusdec)
        pipeline.add(queue)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(alsasink)

        udpsrc.link(capsfilter)
        capsfilter.link(rtpjitterbuffer)
        rtpjitterbuffer.link(rtpopusdepay)
        rtpopusdepay.link(opusdec)
        opusdec.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(alsasink)

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the current camera resolution as a tuple (width, height)."""
        return (self._resolution.value[0], self._resolution.value[1])

    @property
    def framerate(self) -> int:
        """Get the current camera framerate."""
        return self._resolution.value[2]

    def _configure_video(
        self, cam_path: str, pipeline: Gst.Pipeline, webrtcsink: Gst.Element
    ) -> None:
        self._logger.debug("Configuring video")

        if cam_path == "":
            self._logger.warning("Recording pipeline set without camera.")
        elif cam_path == "imx708":  # Reachy Mini wireless / RPI camera
            camsrc = Gst.ElementFactory.make("libcamerasrc")

            caps = Gst.Caps.from_string(
                f"video/x-raw,width={self.resolution[0]},height={self.resolution[1]},framerate={self.framerate}/1,format=YUY2,colorimetry=bt709,interlace-mode=progressive"
            )
            capsfilter = Gst.ElementFactory.make("capsfilter")
            capsfilter.set_property("caps", caps)
            queue = Gst.ElementFactory.make("queue")
            v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
            extra_controls_structure = Gst.Structure.new_empty("extra-controls")
            extra_controls_structure.set_value("repeat_sequence_header", 1)
            extra_controls_structure.set_value("video_bitrate", 5_000_000)
            v4l2h264enc.set_property("extra-controls", extra_controls_structure)
            caps_h264 = Gst.Caps.from_string(
                "video/x-h264,stream-format=byte-stream,alignment=au,level=(string)4"
            )
            capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
            capsfilter_h264.set_property("caps", caps_h264)

            if not all(
                [
                    camsrc,
                    capsfilter,
                    queue,
                    v4l2h264enc,
                    capsfilter_h264,
                ]
            ):
                raise RuntimeError("Failed to create GStreamer video elements")

            pipeline.add(camsrc)
            pipeline.add(capsfilter)
            pipeline.add(queue)
            pipeline.add(v4l2h264enc)
            pipeline.add(capsfilter_h264)
            camsrc.link(capsfilter)
            capsfilter.link(queue)
            queue.link(v4l2h264enc)
            v4l2h264enc.link(capsfilter_h264)
            capsfilter_h264.link(webrtcsink)
        else:
            camsrc = Gst.ElementFactory.make("v4l2src")
            camsrc.set_property("device", cam_path)

            jpegdec = Gst.ElementFactory.make("vajpegdec")  # todo support more?
            if jpegdec is None:
                jpegdec = Gst.ElementFactory.make("jpegdec")
            queue = Gst.ElementFactory.make("queue")

            if not all(
                [
                    camsrc,
                    queue,
                    jpegdec,
                ]
            ):
                raise RuntimeError("Failed to create GStreamer video elements")

            pipeline.add(camsrc)
            pipeline.add(queue)
            pipeline.add(jpegdec)
            camsrc.link(jpegdec)
            jpegdec.link(queue)
            queue.link(webrtcsink)

    def _configure_audio(self, pipeline: Gst.Pipeline, webrtcsink: Gst.Element) -> None:
        self._logger.debug("Configuring audio")
        alsasrc = Gst.ElementFactory.make("alsasrc")
        alsasrc.set_property("device", f"hw:{self._id_audio_card},0")
        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")
        opusenc = Gst.ElementFactory.make("opusenc")
        opusenc.set_property("audio-type", "restricted-lowdelay")
        caps = Gst.Caps.from_string("audio/x-opus,channels=1,rate=48000")
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)

        if not all(
            [
                alsasrc,
                queue,
                audioconvert,
                audioresample,
                opusenc,
                capsfilter,
            ]
        ):
            raise RuntimeError("Failed to create GStreamer audio elements")

        pipeline.add(alsasrc)
        pipeline.add(queue)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(opusenc)
        pipeline.add(capsfilter)

        alsasrc.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(opusenc)
        opusenc.link(capsfilter)
        capsfilter.link(webrtcsink)

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error(f"Error: {err} {debug}")
            return False

        else:
            # self._logger.warning(f"Unhandled message type: {t}")
            pass

        return True

    def start(self) -> None:
        """Start the WebRTC pipeline."""
        self._logger.debug("Starting WebRTC")
        self._pipeline_sender.set_state(Gst.State.PLAYING)
        self._pipeline_receiver.set_state(Gst.State.PLAYING)

    def pause(self) -> None:
        """Pause the WebRTC pipeline."""
        self._logger.debug("Pausing WebRTC")
        self._pipeline_sender.set_state(Gst.State.PAUSED)
        self._pipeline_receiver.set_state(Gst.State.PAUSED)

    def stop(self) -> None:
        """Stop the WebRTC pipeline."""
        self._logger.debug("Stopping WebRTC")

        self._pipeline_sender.set_state(Gst.State.NULL)
        self._pipeline_receiver.set_state(Gst.State.NULL)

    def get_video_device(self) -> Tuple[str, Optional[CameraSpecs]]:
        """Use Gst.DeviceMonitor to find the unix camera path /dev/videoX.

        Returns the device path (e.g., '/dev/video2'), or '' if not found.
        """
        monitor = Gst.DeviceMonitor()
        monitor.add_filter("Video/Source")
        monitor.start()

        cam_names = ["Reachy", "Arducam_12MP", "imx708"]

        devices = monitor.get_devices()
        for cam_name in cam_names:
            for device in devices:
                name = device.get_display_name()
                device_props = device.get_properties()

                if cam_name in name:
                    if device_props and device_props.has_field("api.v4l2.path"):
                        device_path = device_props.get_string("api.v4l2.path")
                        camera_specs = (
                            cast(CameraSpecs, ArducamSpecs)
                            if cam_name == "Arducam_12MP"
                            else cast(CameraSpecs, ReachyMiniLiteCamSpecs)
                        )
                        self._logger.debug(f"Found {cam_name} camera at {device_path}")
                        monitor.stop()
                        return str(device_path), camera_specs
                    elif cam_name == "imx708":
                        camera_specs = cast(CameraSpecs, ReachyMiniWirelessCamSpecs)
                        self._logger.debug(f"Found {cam_name} camera")
                        monitor.stop()
                        return cam_name, camera_specs
        monitor.stop()
        self._logger.warning("No camera found.")
        return "", None


if __name__ == "__main__":
    import os
    import time

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    os.environ["GST_DEBUG"] = "3"
    webrtc = GstWebRTC(localhost=True, log_level="DEBUG")
    webrtc.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("User interrupted")
    finally:
        webrtc.stop()
