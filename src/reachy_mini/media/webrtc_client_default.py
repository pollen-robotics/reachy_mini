"""Default WebRTC client implementation.

The class is a client for the webrtc server hosted on the Reachy Mini Wireless robot.
"""

import asyncio
import logging
import queue
import threading
from typing import Optional

import numpy as np
import numpy.typing as npt
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from gst_signalling.gst_abstract_role import GstSession
from gst_signalling.gst_consumer import GstSignallingConsumer

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.camera_base import CameraBase


class AudioTrack(MediaStreamTrack):
    """A tactile stream track that feeds a audio rendering engine."""

    kind = "audio"

    def __init__(self, track):
        """Initialize the AudioTrack with a given track."""
        super().__init__()  # don't forget this!
        self.track = track
        self.logger = logging.getLogger(__name__)
        self.logger.info("Audio track created")

    def stop(self):
        """Stop the audio track."""
        self.logger.info("Audio stop")

    async def recv(self):
        """Receive an audio frame asynchronously."""
        frame = await self.track.recv()
        self.logger.debug(f"Audio frame: {frame}")
        """
        if self.first_start:
            self.first_start = False
            device_id = 15# GetDefaultDeviceID()
            if device_id == -1:
                self.logger.error("Audio device not found")
            else:
                self.logger.info("Start renderer")
                if frame.format.name != "s16":
                    self.logger.warning("audio sample format not supported")
                self.audioR.run(
                    sampling_rate=frame.rate, device_id=device_id, blocksize=frame.samples, dtype='s16')
        data = frame.to_ndarray(format="s16")
        if frame.format.is_planar is False:
            data = data.reshape((frame.samples, 2))  # len(data) == samples * 2
            # data = np.array([data[:, 0], data[:, 1]])
            self.audioR.fill_data(data)
        else:
            self.logger.warning("Planar data not supported")
    """


class VideoTrack(MediaStreamTrack):
    """A tactile stream track that feeds a video rendering engine."""

    kind = "video"

    def __init__(self, track):
        """Initialize the VideoTrack with a given track."""
        super().__init__()  # don't forget this!
        self.track = track
        self.logger = logging.getLogger(__name__)
        self.logger.info("Video track created")
        # Initialise le thread d'affichage une seule fois (pour tous les objets VideoTrack)
        # if VideoTrack.image_queue is None:
        #    VideoTrack.image_queue = queue.Queue(maxsize=2)
        self.image_queue = queue.Queue(maxsize=2)

    def stop(self):
        """Stop the video track."""
        self.logger.info("Video stop")

    async def recv(self):
        """Receive a video frame asynchronously."""
        frame = await self.track.recv()
        self.logger.debug(f"Video frame: {frame}")
        # Décodage de la frame H264 avec OpenCV
        img = frame.to_ndarray(format="bgr24")
        self.logger.debug(f"Decoded frame shape: {img.shape}")

        # Envoie l'image au thread d'affichage
        try:
            self.image_queue.put_nowait(img)
        except queue.Full:
            self.logger.warning("Image queue is full, dropping frame")
        # return frame

    def get_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Get the latest video frame from the queue.

        Returns:
            Optional[npt.NDArray[np.uint8]]: The latest video frame in BGR format, or None if no frame is available.

        """
        try:
            frame = self.image_queue.get_nowait()
            return frame
        except queue.Empty:
            return None


class VideoClient:
    """Aiortc WebRTC client implementation."""

    def __init__(self, signalling_url: str, signalling_port: int, peer_id: str):
        """Initialize the AioRTC WebRTC client."""
        self.logger = logging.getLogger(__name__)
        self.client = GstSignallingConsumer(signalling_url, signalling_port, peer_id)
        self.media = MediaBlackhole()
        self.audio_track = None
        self.video_track = None

        @self.client.on("new_session")  # type: ignore[misc]
        def on_new_session(session: GstSession) -> None:
            pc = session.pc

            @pc.on("track")
            async def on_track(track):
                self.logger.info("Receiving %s" % track.kind)
                if track.kind == "audio":
                    if self.audio_track is not None:
                        self.logger.warning("Already have an audio track, ignoring")
                        return
                    self.audio_track = AudioTrack(track)
                    self.media.addTrack(self.audio_track)
                    await self.media.start()
                elif track.kind == "video":
                    if self.video_track is not None:
                        self.logger.warning("Already have a video track, ignoring")
                        return
                    self.video_track = VideoTrack(track)
                    self.media.addTrack(self.video_track)
                    await self.media.start()

        def get_frame(self) -> Optional[npt.NDArray[np.uint8]]:
            """Get the latest video frame from the video track.

            Returns:
                Optional[npt.NDArray[np.uint8]]: The latest video frame in BGR format, or None if no frame is available.

            """
            if self.video_track is None:
                self.logger.warning("No video track available")
                return None
            return self.video_track.get_frame()

    async def start(self) -> None:
        """Start the WebRTC client and connect to the server."""
        await self.client.connect()

    async def serve4ever(self) -> None:
        """Keep the client running indefinitely."""
        await self.client.consume()


class DefaultWebRTCClient(CameraBase, AudioBase):
    """Aiortc WebRTC client implementation."""

    def __init__(
        self,
        log_level: str = "INFO",
        peer_id: str = "",
        signaling_host: str = "",
        signaling_port: int = 8443,
    ):
        """Initialize the GStreamer WebRTC client."""
        super().__init__(log_level=log_level)
        self._client = VideoClient(signaling_host, signaling_port, peer_id)
        self._thread_loop: Optional[threading.Thread] = None

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        super().__del__()

    def open(self) -> None:
        """Open the video stream in a background thread."""

        def run_client(client):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def runner():
                await client.start()
                await client.serve4ever()

            loop.run_until_complete(runner())

        self._thread_loop = threading.Thread(
            target=run_client, args=(self._client,), daemon=True
        )
        self._thread_loop.start()

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read a sample from the audio card. Returns the sample or None if error.

        Returns:
            Optional[npt.NDArray[np.float32]]: The captured sample in raw format, or None if error.

        """
        return None  # Not implemented yet

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from the camera. Returns the frame or None if error.

        Returns:
            Optional[npt.NDArray[np.uint8]]: The captured frame in BGR format, or None if error.

        """
        return self._client.get_frame()

    def close(self) -> None:
        """Stop the pipeline."""
        # self._loop.quit()
        pass

    def start_recording(self) -> None:
        """Open the audio card using GStreamer."""
        pass  # already started in open()

    def stop_recording(self) -> None:
        """Release the camera resource."""
        pass  # managed in close()

    def start_playing(self) -> None:
        """Open the audio output using GStreamer."""
        pass

    def stop_playing(self) -> None:
        """Stop playing audio and release resources."""
        pass

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output device."""
        pass

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play.

        """
        self.logger.warning("Audio playback not implemented in WebRTC client.")
