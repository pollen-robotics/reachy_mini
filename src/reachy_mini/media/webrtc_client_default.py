"""Default WebRTC client implementation.

The class is a client for the webrtc server hosted on the Reachy Mini Wireless robot.
"""

import asyncio
import fractions
import logging
import queue
import random
import socket
import threading
from typing import Optional

import numpy as np
import numpy.typing as npt
from aiortc import MediaStreamTrack
from aiortc.codecs.opus import OpusEncoder
from aiortc.contrib.media import MediaBlackhole
from av import AudioFrame, AudioResampler
from gst_signalling.gst_abstract_role import GstSession
from gst_signalling.gst_consumer import GstSignallingConsumer

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.camera_base import CameraBase


class AudioTrack(MediaStreamTrack):
    """A tactile stream track that feeds a audio rendering engine."""

    kind = "audio"
    QUEUE_MAXSIZE = 200

    def __init__(self, track, sample_rate: int, log_level: str = "INFO"):
        """Initialize the AudioTrack with a given track."""
        super().__init__()  # don't forget this!
        self.track = track
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.debug("Audio track created")
        self.audio_queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._resampler = AudioResampler(
            format="flt", layout="stereo", rate=sample_rate
        )

    def stop(self):
        """Stop the audio track."""
        self.logger.debug("Audio stop")

    async def recv(self):
        """Receive an audio frame asynchronously."""
        frame = await self.track.recv()
        self.logger.debug(f"Audio frame: {frame}")

        try:
            resampled_frame = self._resampler.resample(frame)
            if len(resampled_frame) > 1:
                self.logger.warning(f"Resampled frame: {resampled_frame}")
            data = resampled_frame[0].to_ndarray()

            if resampled_frame[0].format.is_planar is False:
                data = data.reshape((resampled_frame[0].samples, 2))
            else:
                self.logger.warning("Planar data not supported")

            self.audio_queue.put_nowait(data)
        except queue.Full:
            self.logger.debug("Audio queue is full, dropping frame")
        except Exception as e:
            self.logger.error(f"Error putting audio data into queue: {e}")

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Get the latest audio sample from the queue.

        Returns:
            Optional[npt.NDArray[np.float32]]: The latest audio sample, or None if no sample is available.

        """
        try:
            data = self.audio_queue.get_nowait()
            return data
        except queue.Empty:
            return None


class VideoTrack(MediaStreamTrack):
    """A tactile stream track that feeds a video rendering engine."""

    kind = "video"

    def __init__(self, track, log_level: str = "INFO"):
        """Initialize the VideoTrack with a given track."""
        super().__init__()  # don't forget this!
        self.track = track
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.debug("Video track created")

        self._latest_image = None
        self._image_lock = threading.Lock()

    def stop(self):
        """Stop the video track."""
        self.logger.debug("Video stop")

    async def recv(self):
        """Receive a video frame asynchronously."""
        frame = await self.track.recv()
        self.logger.debug(f"Video frame: {frame}")
        img = frame.to_ndarray(format="bgr24")
        self.logger.debug(f"Decoded frame shape: {img.shape}")
        with self._image_lock:
            self._latest_image = img

    def get_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Get the latest video frame.

        Returns:
            Optional[npt.NDArray[np.uint8]]: The latest video frame in BGR format, or None if no frame is available.

        """
        if self._image_lock.acquire(blocking=False):
            try:
                if self._latest_image is not None:
                    return self._latest_image.copy()
                else:
                    return None
            finally:
                self._image_lock.release()
        else:
            return None


class VideoClient:
    """Aiortc WebRTC client implementation."""

    def __init__(
        self,
        signalling_url: str,
        signalling_port: int,
        peer_id: str,
        sample_rate: int,
        log_level: str = "INFO",
    ):
        """Initialize the AioRTC WebRTC client."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._log_level = log_level
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
                    self.audio_track = AudioTrack(
                        track, sample_rate=sample_rate, log_level=self._log_level
                    )
                    self.media.addTrack(self.audio_track)
                    await self.media.start()
                elif track.kind == "video":
                    if self.video_track is not None:
                        self.logger.warning("Already have a video track, ignoring")
                        return
                    self.video_track = VideoTrack(track, log_level=self._log_level)
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

    def get_audio_sample(self) -> Optional[npt.NDArray[np.int16]]:
        """Get the latest audio sample from the audio track.

        Returns:
            Optional[npt.NDArray[np.int16]]: The latest audio sample, or None if no sample is available.

        """
        if self.audio_track is None:
            self.logger.warning("No audio track available")
            return None

        return self.audio_track.get_audio_sample()

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
        CameraBase.__init__(self, log_level=log_level)
        AudioBase.__init__(self, log_level=log_level)
        self.logger.info("Initializing Default WebRTC Client")
        self._signaling_host = signaling_host
        self._client = VideoClient(
            signaling_host,
            signaling_port,
            peer_id,
            AudioBase.SAMPLE_RATE,
            log_level=log_level,
        )
        self._thread_loop: Optional[threading.Thread] = None
        self._playback_pts = 0
        self._rtp_seq = 0
        self._rtp_ssrc = 0
        self._sock_playback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        AudioBase.__del__(self)

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
        return self._client.get_audio_sample()

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
        self._rtp_seq = 0
        self._playback_pts = 0
        self._rtp_timestamp = 0
        self._rtp_ssrc = random.getrandbits(32)
        print(f"Playback SSRC: {self._rtp_ssrc}")
        self._sock_playback.connect((self._signaling_host, 5000))

    def stop_playing(self) -> None:
        """Stop playing audio and release resources."""
        self._sock_playback.close()

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output device."""
        print(data.dtype)
        data = data.reshape((1, -1))
        data_s16 = (data * 32767.0).astype(np.int16)
        samples = 1  # int(AudioBase.SAMPLE_RATE * AUDIO_PTIME)
        frame = AudioFrame.from_ndarray(data_s16, format="s16", layout="mono")
        frame.rate = AudioBase.SAMPLE_RATE
        frame.time_base = fractions.Fraction(1, AudioBase.SAMPLE_RATE)
        frame.pts = self._playback_pts
        self._playback_pts += samples

        print(f"Frame rate: {frame.rate}")
        encoder = OpusEncoder()
        encoded = encoder.encode(frame)
        # print(f"Opus encoded audio length: {len(encoded[0])} bytes")
        # print(f"Opus encoded audio: {encoded}")
        # Now you can use 'frame' for further processing (encoding, streaming, etc.)
        # Example: print(frame)
        print(
            f"Created AudioFrame: samples={frame.samples}, sample_rate={frame.rate}, layout={frame.layout}"
        )
        print(
            f"  PTS: {frame.pts}, encoded packets: {(encoded[1])}, pts next: {self._playback_pts}"
        )
        for i, p in enumerate(encoded[0]):
            print(f"Encoded packet length: {len(p)} bytes")
            """
            rtp = RtpPacket(
                payload_type=96,
                sequence_number=self._rtp_seq,
                timestamp=self._rtp_timestamp,
                ssrc=self._rtp_ssrc,
                payload=p,
            )
            packet_bytes = rtp.serialize()

            self._sock_playback.send(packet_bytes)
            self._rtp_seq += 1
            self._rtp_timestamp += 960 // len(encoded[0])
            """

            self._sock_playback.send(p)

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play.

        """
        self.logger.warning("Audio playback not implemented in WebRTC client.")
