"""Media Manager.

Provides camera and audio access based on the selected backend.

This module offers a unified interface for managing both camera and audio
devices with support for multiple backends. It simplifies the process of
initializing, configuring, and using media devices across different
platforms and use cases.

"""

import logging
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.camera_base import CameraBase

# actual backends are dynamically imported


class MediaBackend(Enum):
    """Media backends.

    Enumeration of available media backends that can be used with MediaManager.
    Each backend provides different capabilities and performance characteristics.

    Attributes:
        NO_MEDIA: No media devices - useful for headless operation or when
                 media devices are not needed.
        DEFAULT: Default backend using OpenCV for video and SoundDevice for audio.
                Cross-platform and widely compatible.
        DEFAULT_NO_VIDEO: SoundDevice audio only - for audio processing without video.
        GSTREAMER: GStreamer-based media backend with advanced video and audio
                  processing capabilities.
        GSTREAMER_NO_VIDEO: GStreamer audio only - for advanced audio processing
                           without video.
        WEBRTC: WebRTC-based media backend for real-time communication and
               streaming applications.

    Example:
        ```python
        from reachy_mini.media.media_manager import MediaBackend

        # Select the appropriate backend for your use case
        backend = MediaBackend.DEFAULT  # Cross-platform default
        # backend = MediaBackend.GSTREAMER  # Advanced features on Linux
        # backend = MediaBackend.WEBRTC  # Real-time streaming
        # backend = MediaBackend.NO_MEDIA  # Headless operation
        ```

    """

    NO_MEDIA = "no_media"
    DEFAULT = "default"
    DEFAULT_NO_VIDEO = "default_no_video"
    GSTREAMER = "gstreamer"
    GSTREAMER_NO_VIDEO = "gstreamer_no_video"
    WEBRTC = "webrtc"


class MediaManager:
    """Media Manager for handling camera and audio devices.

        This class provides a unified interface for managing both camera and audio
    devices across different backends. It handles initialization, configuration,
    and cleanup of media resources.

    Attributes:
            logger (logging.Logger): Logger instance for media-related messages.
            backend (MediaBackend): The selected media backend.
            camera (Optional[CameraBase]): Camera device instance.
            audio (Optional[AudioBase]): Audio device instance.

    """

    def __init__(
        self,
        backend: MediaBackend = MediaBackend.DEFAULT,
        log_level: str = "INFO",
        use_sim: bool = False,
        signalling_host: str = "localhost",
    ) -> None:
        """Initialize the media manager.

        Args:
            backend (MediaBackend): The media backend to use. Default is DEFAULT.
            log_level (str): Logging level for media operations.
                          Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
                          Default: 'INFO'.
            use_sim (bool): Whether to use simulation mode (for testing).
                          Default: False.
            signalling_host (str): Host address for WebRTC signalling server.
                                 Only used with WEBRTC backend.
                                 Default: 'localhost'.

        Note:
            The constructor initializes the selected media backend and sets up
            the appropriate camera and audio devices based on the backend choice.

        Available backends:
            - NO_MEDIA: No media devices (useful for headless operation)
            - DEFAULT: OpenCV + SoundDevice (cross-platform default)
            - DEFAULT_NO_VIDEO: SoundDevice only (audio without video)
            - GSTREAMER: GStreamer-based media (advanced features)
            - GSTREAMER_NO_VIDEO: GStreamer audio only
            - WEBRTC: WebRTC-based media for real-time communication

        Example usage:
            ```python
            from reachy_mini.media.media_manager import MediaManager, MediaBackend

            # Initialize with default backend
            media = MediaManager(backend=MediaBackend.DEFAULT)

            # Capture a frame
            frame = media.get_frame()
            if frame is not None:
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)

            # Play a sound
            media.play_sound("/path/to/sound.wav")

            # Clean up
            media.close()
            ```

        """
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
                self._init_camera(use_sim, log_level)
                self._init_audio(log_level)
            case MediaBackend.DEFAULT_NO_VIDEO:
                self.logger.info("Using default media backend (SoundDevice only).")
                self._init_audio(log_level)
            case MediaBackend.GSTREAMER:
                self.logger.info("Using GStreamer media backend.")
                self._init_camera(use_sim, log_level)
                self._init_audio(log_level)
            case MediaBackend.GSTREAMER_NO_VIDEO:
                self.logger.info("Using GStreamer audio backend.")
                self._init_audio(log_level)
            case MediaBackend.WEBRTC:
                self.logger.info("Using WebRTC GStreamer backend.")
                self._init_webrtc(log_level, signalling_host, 8443)
                # self._init_audio(log_level)
            case _:
                raise NotImplementedError(f"Media backend {backend} not implemented.")

    def close(self) -> None:
        """Close the media manager and release resources.

        This method should be called when the media manager is no longer needed
        to properly clean up and release all media resources. It stops any ongoing
        audio recording/playback and closes the camera device.

        Note:
            After calling this method, the media manager can be reused by calling
            the appropriate initialization methods again, but it's generally
            recommended to create a new MediaManager instance if needed.

        Example:
            ```python
            media = MediaManager()
            try:
                # Use media devices
                frame = media.get_frame()
            finally:
                media.close()
            ```

        """
        if self.camera is not None:
            self.camera.close()
        if self.audio is not None:
            self.audio.stop_recording()
            self.audio.stop_playing()

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        self.close()

    def _init_camera(
        self,
        use_sim: bool,
        log_level: str,
    ) -> None:
        """Initialize the camera."""
        self.logger.debug("Initializing camera...")
        if self.backend == MediaBackend.DEFAULT:
            self.logger.info("Using OpenCV camera backend.")
            from reachy_mini.media.camera_opencv import OpenCVCamera

            self.camera = OpenCVCamera(log_level=log_level)
            if use_sim:
                self.camera.open(udp_camera="udp://@127.0.0.1:5005")
            else:
                self.camera.open()
        elif self.backend == MediaBackend.GSTREAMER:
            self.logger.info("Using GStreamer camera backend.")
            from reachy_mini.media.camera_gstreamer import GStreamerCamera

            self.camera = GStreamerCamera(log_level=log_level)
            self.camera.open()
            # Todo: use simulation with gstreamer?

        else:
            raise NotImplementedError(f"Camera backend {self.backend} not implemented.")

    def get_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Get a frame from the camera.

        Returns:
            Optional[npt.NDArray[np.uint8]]: The captured BGR frame as a numpy array
            with shape (height, width, 3), or None if the camera is not available
            or an error occurred.

            The image is in BGR format (OpenCV convention) and can be directly
            used with OpenCV functions or converted to RGB if needed.

        Note:
            This method returns None if the camera is not initialized or if
            there's an error capturing the frame. Always check the return value
            before using the frame.

        Example:
            ```python
            frame = media.get_frame()
            if frame is not None:
                # Process the frame
                cv2.imshow("Camera", frame)
                cv2.waitKey(1)

                # Convert to RGB if needed
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ```

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
        elif (
            self.backend == MediaBackend.GSTREAMER
            or self.backend == MediaBackend.GSTREAMER_NO_VIDEO
        ):
            self.logger.info("Using GStreamer audio backend.")
            from reachy_mini.media.audio_gstreamer import GStreamerAudio

            self.audio = GStreamerAudio(log_level=log_level)
        else:
            raise NotImplementedError(f"Audio backend {self.backend} not implemented.")

    def _init_webrtc(
        self, log_level: str, signalling_host: str, signalling_port: int
    ) -> None:
        """Initialize the WebRTC system (not implemented yet)."""
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

    def get_audio_sample(self) -> Optional[bytes | npt.NDArray[np.float32]]:
        """Get an audio sample from the audio device.

        Returns:
            Optional[np.ndarray]: The recorded audio sample, or None if no data is available.

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return None
        return self.audio.get_audio_sample()

    def get_input_audio_samplerate(self) -> int:
        """Get the input samplerate of the audio device."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return -1
        return self.audio.get_input_audio_samplerate()

    def get_output_audio_samplerate(self) -> int:
        """Get the output samplerate of the audio device."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return -1
        return self.audio.get_output_audio_samplerate()

    def get_input_channels(self) -> int:
        """Get the number of input channels of the audio device."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return -1
        return self.audio.get_input_channels()

    def get_output_channels(self) -> int:
        """Get the number of output channels of the audio device."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return -1
        return self.audio.get_output_channels()

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

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to the output device.

        Args:
            data (npt.NDArray[np.float32]): The audio data to push to the output device (mono format).

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return

        if data.ndim > 2 or data.ndim == 0:
            self.logger.warning(
                f"Audio samples arrays must have at most 2 dimensions and at least 1 dimension, got {data.ndim}"
            )
            return

        # Transpose data to match sounddevice channels last convention
        if data.ndim == 2 and data.shape[1] > data.shape[0]:
            data = data.T

        # Fit data to match output stream channels
        output_channels = self.get_output_channels()

        # Mono input to multiple channels output : duplicate to fit
        if data.ndim == 1 and output_channels > 1:
            data = np.column_stack((data,) * output_channels)
        # Lower channels input to higher channels output : reduce to mono and duplicate to fit
        elif data.ndim == 2 and data.shape[1] < output_channels:
            data = np.column_stack((data[:, 0],) * output_channels)
        # Higher channels input to lower channels output : crop to fit
        elif data.ndim == 2 and data.shape[1] > output_channels:
            data = data[:, :output_channels]

        self.audio.push_audio_sample(data)

    def stop_playing(self) -> None:
        """Stop playing audio."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return
        self.audio.stop_playing()

    def get_DoA(self) -> tuple[float, bool] | None:
        """Get the Direction of Arrival (DoA) from the microphone array.

        Returns:
            tuple[float, bool] | None: A tuple (angle_radians, speech_detected),
            or None if the audio system is not available.

        """
        if self.audio is None:
            return None
        return self.audio.get_DoA()
