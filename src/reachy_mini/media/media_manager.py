"""Media Manager.

Provides camera and audio access based on the selected backedn
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.camera_base import CameraBase
from reachy_mini.media.camera_constants import CameraResolution

# actual backends are dynamically imported


class MediaBackend(Enum):
    """Media backends."""

    NO_MEDIA = "no_media"
    DEFAULT = "default"
    DEFAULT_NO_VIDEO = "default_no_video"
    GSTREAMER = "gstreamer"
    WEBRTC = "webrtc"


class MediaManager:
    """Abstract class for opening and managing audio devices."""

    def __init__(
        self,
        backend: MediaBackend = MediaBackend.DEFAULT,
        log_level: str = "INFO",
        use_sim: bool = False,
        resolution: CameraResolution = CameraResolution.R1280x720,
        signalling_host: str = "localhost",
    ) -> None:
        """Initialize the audio device."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.backend = backend
        self.camera: Optional[CameraBase] = None
        self.audio: Optional[AudioBase] = None

        match backend:
            case MediaBackend.NO_MEDIA:
                self.logger.info("No media backend selected.")
            case MediaBackend.DEFAULT:
                self.logger.info("Using default media backend (OpenCV + SoundDevice).")
                self._init_camera(use_sim, log_level, resolution)
                self._init_audio(log_level)
            case MediaBackend.DEFAULT_NO_VIDEO:
                self.logger.info("Using default media backend (SoundDevice only).")
                self._init_audio(log_level)
            case MediaBackend.GSTREAMER:
                self.logger.info("Using GStreamer media backend.")
                self._init_camera(use_sim, log_level, resolution)
                self._init_audio(log_level)
            case MediaBackend.WEBRTC:
                self.logger.info("Using WebRTC GStreamer backend.")
                self._init_webrtc(log_level, signalling_host, 8443)
            case _:
                raise NotImplementedError(f"Media backend {backend} not implemented.")

    def _init_camera(
        self,
        use_sim: bool,
        log_level: str,
        resolution: CameraResolution,
    ) -> None:
        """Initialize the camera."""
        self.logger.debug("Initializing camera...")
        if self.backend == MediaBackend.DEFAULT:
            self.logger.info("Using OpenCV camera backend.")
            from reachy_mini.media.camera_opencv import OpenCVCamera

            self.camera = OpenCVCamera(log_level=log_level, resolution=resolution)
            if use_sim:
                self.camera.open(udp_camera="udp://@127.0.0.1:5005")
            else:
                self.camera.open()
        elif self.backend == MediaBackend.GSTREAMER:
            self.logger.info("Using GStreamer camera backend.")
            from reachy_mini.media.camera_gstreamer import GStreamerCamera

            self.camera = GStreamerCamera(log_level=log_level, resolution=resolution)
            self.camera.open()
            # Todo: use simulation with gstreamer?

        else:
            raise NotImplementedError(f"Camera backend {self.backend} not implemented.")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get a frame from the camera.

        Returns:
            Optional[np.ndarray]: The captured frame, or None if the camera is not available.

        """
        if self.camera is None:
            self.logger.warning("Camera is not initialized.")
            return None
        return self.camera.read()

    def _init_audio(self, log_level: str) -> None:
        """Initialize the audio system."""
        self.logger.debug("Initializing audio...")
        if (
            self.backend == MediaBackend.DEFAULT
            or self.backend == MediaBackend.DEFAULT_NO_VIDEO
        ):
            self.logger.info("Using SoundDevice audio backend.")
            from reachy_mini.media.audio_sounddevice import SoundDeviceAudio

            self.audio = SoundDeviceAudio(log_level=log_level)
        elif self.backend == MediaBackend.GSTREAMER:
            self.logger.info("Using GStreamer audio backend.")
            from reachy_mini.media.audio_gstreamer import GStreamerAudio

            self.audio = GStreamerAudio(log_level=log_level)
        else:
            raise NotImplementedError(f"Audio backend {self.backend} not implemented.")

    def _init_webrtc(
        self, log_level: str, signalling_host: str, signalling_port: int
    ) -> None:
        """Initialize the WebRTC system (not implemented yet)."""
        self.logger.warning("WebRTC media backend is not implemented yet.")
        from gst_signalling.utils import find_producer_peer_id_by_name

        from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient

        peer_id = find_producer_peer_id_by_name(
            signalling_host, signalling_port, "reachymini"
        )

        webrtc_media: GstWebRTCClient = GstWebRTCClient(
            log_level=log_level,
            peer_id=peer_id,
            signaling_host=signalling_host,
            signaling_port=signalling_port,
        )

        self.camera = webrtc_media
        self.audio = webrtc_media  # GstWebRTCClient handles both audio and video
        self.camera.open()

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play.

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return
        self.audio.play_sound(sound_file)

    def start_recording(self) -> None:
        """Start recording audio."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return
        self.audio.start_recording()

    def get_audio_sample(self) -> Optional[np.ndarray]:
        """Get an audio sample from the audio device.

        Returns:
            Optional[np.ndarray]: The recorded audio sample, or None if no data is available.

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return None
        return self.audio.get_audio_sample()

    def get_audio_samplerate(self) -> int:
        """Get the samplerate of the audio device.

        Returns:
            int: The samplerate of the audio device.

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return -1
        return self.audio.get_audio_samplerate()

    def stop_recording(self) -> None:
        """Stop recording audio."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return
        self.audio.stop_recording()

    def start_playing(self) -> None:
        """Start playing audio."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return
        self.audio.start_playing()

    def push_audio_sample(self, data) -> None:
        """Push audio data to the output device.

        Args:
            data: The audio data to push to the output device.

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return
        self.audio.push_audio_sample(data)

    def stop_playing(self) -> None:
        """Stop playing audio."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return
        self.audio.stop_playing()
