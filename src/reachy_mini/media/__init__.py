"""Media module for Reachy Mini robot.

This module provides audio and video capabilities for the Reachy Mini robot,
supporting multiple backends and offering a unified interface for media operations.

Architecture:
    The daemon always owns the physical camera and audio hardware via
    ``webrtc_daemon.py``.  Client-side code selects a backend through
    ``MediaManager``:

    * **LOCAL** – reads camera frames from the daemon's IPC endpoint and opens
      the local audio device directly.  Best for on-device apps.
    * **WEBRTC** – streams camera + audio over WebRTC from the daemon.
      Best for remote clients.
    * **NO_MEDIA** – skips all media initialisation (headless operation).

Key Components:
    - MediaManager: Unified interface for managing audio and video devices
    - CameraBase: Abstract base class for camera implementations
    - AudioBase: Abstract base class for audio implementations

Example usage::

    from reachy_mini.media.media_manager import MediaManager, MediaBackend

    # Create media manager with default (LOCAL) backend
    media = MediaManager(backend=MediaBackend.DEFAULT)

    # Capture video frames
    frame = media.get_frame()

    # Record audio
    media.start_recording()
    samples = media.get_audio_sample()

    # Play sound
    media.play_sound("/path/to/sound.wav")

    # Clean up
    media.close()

For more information on specific components, see:
    - media_manager.py: Media management and backend selection
    - camera_base.py: Camera interface definition
    - audio_base.py: Audio interface definition
    - camera_gstreamer.py: GStreamer IPC camera reader (LOCAL backend)
    - audio_gstreamer.py: GStreamer audio implementation (LOCAL backend)
    - webrtc_client_gstreamer.py: WebRTC client (WEBRTC backend)
    - webrtc_daemon.py: WebRTC daemon/server (runs on the daemon side)
"""
