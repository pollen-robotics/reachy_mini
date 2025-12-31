"""Media module for Reachy Mini robot.

This module provides comprehensive audio and video capabilities for the Reachy Mini robot,
supporting multiple backends and offering a unified interface for media operations.

The media module includes:
- Camera capture and video streaming
- Microphone input and audio recording
- Speaker output and sound playback
- Direction of Arrival (DoA) estimation with ReSpeaker microphone array
- Multiple backend support (OpenCV, GStreamer, SoundDevice, WebRTC)
- Camera calibration and intrinsic parameter access
- Cross-platform compatibility

Key Components:
- MediaManager: Unified interface for managing audio and video devices
- CameraBase: Abstract base class for camera implementations
- AudioBase: Abstract base class for audio implementations
- Multiple backend implementations for different use cases

Example usage:
    >>> from reachy_mini.media.media_manager import MediaManager, MediaBackend
    >>>
    >>> # Create media manager with default backend
    >>> media = MediaManager(backend=MediaBackend.DEFAULT)
    >>>
    >>> # Capture video frames
    >>> frame = media.get_frame()
    >>> if frame is not None:
    ...     cv2.imshow("Camera", frame)
    ...     cv2.waitKey(1)
    >>>
    >>> # Record audio
    >>> media.start_recording()
    >>> samples = media.get_audio_sample()
    >>>
    >>> # Play sound
    >>> media.play_sound("/path/to/sound.wav")
    >>>
    >>> # Clean up
    >>> media.close()

Available backends:
- MediaBackend.DEFAULT: OpenCV + SoundDevice (cross-platform default)
- MediaBackend.GSTREAMER: GStreamer-based media (advanced features)
- MediaBackend.WEBRTC: WebRTC for real-time communication
- MediaBackend.NO_MEDIA: No media devices (headless operation)

For more information on specific components, see:
- media_manager.py: Media management and backend selection
- camera_base.py: Camera interface definition
- audio_base.py: Audio interface definition
- camera_opencv.py: OpenCV camera implementation
- audio_sounddevice.py: SoundDevice audio implementation
"""
