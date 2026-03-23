"""Media Manager.

Provides camera and audio access based on the selected backend.

This module offers a unified interface for managing both camera and audio
devices with support for multiple backends.  It simplifies the process of
initializing, configuring, and using media devices across different
platforms and use cases.

Architecture overview:
    The daemon always owns the physical camera and audio hardware via
    ``GstMediaServer`` (``media_server.py``).  Clients pick one of three strategies:

    * **LOCAL** – read camera frames from the daemon's IPC endpoint
      (``unixfdsrc`` / ``win32ipcvideosrc``) and open the local audio
      device directly via GStreamer.  Best for on-device apps (no
      encode/decode overhead).
    * **WEBRTC** – stream camera + audio over WebRTC from the daemon.
      Best for remote clients.
    * **NO_MEDIA** – skip all media initialisation (headless operation).
"""

import logging
import warnings
from enum import Enum
from typing import Optional, Union

import numpy as np
import numpy.typing as npt


class MediaBackend(Enum):
    """Media backends.

    Attributes:
        NO_MEDIA: No media devices — headless operation.
        LOCAL: GStreamer IPC camera reader + GStreamer local audio.
            Use when the client runs on the same machine as the daemon.
        WEBRTC: WebRTC streaming from the daemon (camera + audio).
            Use when the client is remote.
        DEFAULT: Alias for LOCAL.

    Deprecated values (emit ``FutureWarning``):
        GSTREAMER, GSTREAMER_NO_VIDEO, SOUNDDEVICE_OPENCV,
        SOUNDDEVICE_NO_VIDEO, DEFAULT_NO_VIDEO.

    """

    NO_MEDIA = "no_media"
    LOCAL = "local"
    WEBRTC = "webrtc"

    # Primary alias
    DEFAULT = LOCAL

    # Deprecated aliases — kept so old code keeps working for one release
    GSTREAMER = "gstreamer"
    GSTREAMER_NO_VIDEO = "gstreamer_no_video"
    SOUNDDEVICE_NO_VIDEO = "sounddevice_no_video"
    SOUNDDEVICE_OPENCV = "sounddevice_opencv"
    DEFAULT_NO_VIDEO = GSTREAMER_NO_VIDEO


# Mapping from deprecated enum members to their replacement
_DEPRECATED_BACKENDS: dict[MediaBackend, MediaBackend] = {
    MediaBackend.GSTREAMER: MediaBackend.LOCAL,
    MediaBackend.GSTREAMER_NO_VIDEO: MediaBackend.LOCAL,
    MediaBackend.SOUNDDEVICE_NO_VIDEO: MediaBackend.LOCAL,
    MediaBackend.SOUNDDEVICE_OPENCV: MediaBackend.LOCAL,
}


def _resolve_backend(backend: MediaBackend) -> MediaBackend:
    """Return the canonical backend, emitting a deprecation warning if needed."""
    replacement = _DEPRECATED_BACKENDS.get(backend)
    if replacement is not None:
        warnings.warn(
            f"MediaBackend.{backend.name} is deprecated and will be removed in a "
            f"future release. Use MediaBackend.{replacement.name} instead.",
            FutureWarning,
            stacklevel=3,
        )
        return replacement
    return backend


# -- Type aliases for the concrete backends ----------------------------
# Both GStreamerCamera and GstWebRTCClient expose the same camera API
# (open, read, close, set_resolution, resolution, K, D, …).
# Both GStreamerAudio and GstWebRTCClient expose the same audio API
# (start_recording, get_audio_sample, push_audio_sample, play_sound, …).
# We import them lazily inside the init helpers, but declare the union
# here so the type annotations stay narrow.

from reachy_mini.media.audio_gstreamer import GStreamerAudio  # noqa: E402
from reachy_mini.media.camera_constants import CameraSpecs  # noqa: E402
from reachy_mini.media.camera_gstreamer import GStreamerCamera  # noqa: E402
from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient  # noqa: E402

CameraLike = Union[GStreamerCamera, GstWebRTCClient]
AudioLike = Union[GStreamerAudio, GstWebRTCClient]


class MediaManager:
    """Media Manager for handling camera and audio devices.

    This class provides a unified interface for managing both camera and audio
    devices across different backends.  It handles initialization,
    configuration, and cleanup of media resources.

    Attributes:
        logger: Logger instance for media-related messages.
        backend: The selected media backend (after deprecation resolution).
        camera: Camera device instance, or ``None``.
        audio: Audio device instance, or ``None``.

    """

    def __init__(
        self,
        backend: MediaBackend = MediaBackend.DEFAULT,
        log_level: str = "INFO",
        signalling_host: str = "localhost",
        camera_specs: Optional[CameraSpecs] = None,
        daemon_url: str = "",
    ) -> None:
        """Initialize the media manager.

        Args:
            backend: The media backend to use.  Default is ``LOCAL``.
            log_level: Logging level for media operations.
            signalling_host: Host address for WebRTC signalling server.
                Only used with the ``WEBRTC`` backend.
            camera_specs: Camera specifications detected by the daemon.
                When ``None`` the concrete camera class will fall back to
                ``ReachyMiniLiteCamSpecs`` with a warning.
            daemon_url: Base URL of the daemon's HTTP API
                (e.g. ``"http://reachy-mini.local:8000"``).  Only used
                with the ``WEBRTC`` backend for remote sound playback
                and file management.

        Example::

            from reachy_mini.media.media_manager import MediaManager, MediaBackend

            media = MediaManager(backend=MediaBackend.DEFAULT)
            frame = media.get_frame()
            media.close()

        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.backend = _resolve_backend(backend)
        self.camera: Optional[CameraLike] = None
        self.audio: Optional[AudioLike] = None
        self._daemon_url = daemon_url

        match self.backend:
            case MediaBackend.NO_MEDIA:
                self.logger.info("No media backend selected.")
            case MediaBackend.LOCAL:
                self.logger.info(
                    "Using LOCAL backend (GStreamer IPC camera + GStreamer audio)."
                )
                self._init_camera(log_level, camera_specs)
                self._init_audio(log_level)
            case MediaBackend.WEBRTC:
                self.logger.info("Using WebRTC streaming backend.")
                self._init_webrtc(
                    log_level, signalling_host, 8443, camera_specs, daemon_url
                )
            case _:
                raise NotImplementedError(f"Media backend {backend} not implemented.")

    def close(self) -> None:
        """Close the media manager and release resources."""
        if self.camera is not None:
            self.camera.close()
            del self.camera
            self.camera = None
        if self.audio is not None:
            self.audio.stop_recording()
            self.audio.stop_playing()
            self.audio.cleanup()
            del self.audio
            self.audio = None

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        self.close()

    def _init_camera(
        self, log_level: str, camera_specs: Optional[CameraSpecs] = None
    ) -> None:
        """Initialize the camera via local IPC."""
        self.logger.debug("Initializing camera (LOCAL IPC reader)...")
        self.camera = GStreamerCamera(log_level=log_level, camera_specs=camera_specs)
        self.camera.open()

    def _init_audio(self, log_level: str) -> None:
        """Initialize the audio system via GStreamer."""
        self.logger.debug("Initializing audio (GStreamer)...")
        self.audio = GStreamerAudio(log_level=log_level)

    def _init_webrtc(
        self,
        log_level: str,
        signalling_host: str,
        signalling_port: int,
        camera_specs: Optional[CameraSpecs] = None,
        daemon_url: str = "",
    ) -> None:
        """Initialize the WebRTC client (camera + audio)."""
        from reachy_mini.media.webrtc_utils import find_producer_peer_id_by_name

        peer_id = find_producer_peer_id_by_name(
            signalling_host, signalling_port, "reachymini"
        )

        webrtc_media = GstWebRTCClient(
            log_level=log_level,
            peer_id=peer_id,
            signaling_host=signalling_host,
            signaling_port=signalling_port,
            camera_specs=camera_specs,
        )
        # Auto-derive daemon URL from signalling host when not provided.
        webrtc_media.daemon_url = daemon_url or f"http://{signalling_host}:8000"

        self.camera = webrtc_media
        self.audio = webrtc_media  # GstWebRTCClient handles both audio and video
        self.camera.open()

    def get_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Get a frame from the camera.

        Returns:
            The captured BGR frame as a numpy array with shape
            ``(height, width, 3)``, or ``None`` if the camera is not
            available.

        """
        if self.camera is None:
            self.logger.warning("Camera is not initialized.")
            return None
        return self.camera.read()

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file.

        Args:
            sound_file: Path to the sound file to play.

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return
        self.audio.play_sound(sound_file)

    def upload_sound(self, sound_file: str) -> str:
        """Upload a local sound file to the daemon's temporary directory.

        Only meaningful for the ``WEBRTC`` backend — when using the
        ``LOCAL`` backend the file is already accessible so the path is
        returned as-is.

        Args:
            sound_file: Local path to the sound file.

        Returns:
            The path of the file on the daemon.

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return sound_file
        return self.audio.upload_sound(sound_file)

    def list_sounds(self) -> list[str]:
        """List sound files in the daemon's temporary sound directory.

        Returns:
            A list of filenames.  Empty for the ``LOCAL`` backend.

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return []
        return self.audio.list_sounds()

    def delete_sound(self, filename: str) -> bool:
        """Delete a sound file from the daemon's temporary sound directory.

        Args:
            filename: Name of the file to delete (not a path).

        Returns:
            ``True`` if the file was deleted, ``False`` otherwise.

        """
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return False
        return self.audio.delete_sound(filename)

    def start_recording(self) -> None:
        """Start recording audio."""
        if self.audio is None:
            self.logger.warning("Audio system is not initialized.")
            return
        self.audio.start_recording()

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Get an audio sample from the audio device.

        Returns:
            The recorded audio sample, or ``None`` if no data is available.

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
            data: The audio data to push (mono format).

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
            A tuple ``(angle_radians, speech_detected)``, or ``None`` if the
            audio system is not available.

        """
        if self.audio is None:
            return None
        return self.audio.get_DoA()
