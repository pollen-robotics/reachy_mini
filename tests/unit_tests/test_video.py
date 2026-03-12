import time
from typing import cast

import numpy as np
import pytest

from reachy_mini.daemon.utils import is_local_camera_available
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    MujocoCameraSpecs,
    ReachyMiniLiteCamSpecs,
)
from reachy_mini.media.media_manager import MediaBackend, MediaManager

SIGNALING_HOST = "reachy-mini.local"

# Video-capable backends to test
VIDEO_BACKENDS = [
    pytest.param(MediaBackend.LOCAL),
    pytest.param(MediaBackend.WEBRTC, marks=pytest.mark.wireless),
]


@pytest.mark.video
def test_is_local_camera_available(ipc_video_source: CameraSpecs) -> None:
    """Test that is_local_camera_available() detects the IPC endpoint.

    The ipc_video_source fixture creates a videotestsrc → IPC sink pipeline
    (unixfdsink on Linux/macOS, win32ipcvideosink on Windows).
    While it's running, ``is_local_camera_available()`` must return True.
    """
    assert is_local_camera_available(), (
        "is_local_camera_available() returned False while the IPC fixture is running"
    )


@pytest.mark.video
def test_get_frame_exists(ipc_video_source: CameraSpecs) -> None:
    """Test that a frame can be retrieved from the LOCAL IPC camera."""
    media = MediaManager(
        backend=MediaBackend.LOCAL,
        camera_specs=ipc_video_source,
    )
    time.sleep(1)
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert (
        frame.shape[0] == media.camera.resolution[1]
        and frame.shape[1] == media.camera.resolution[0]
    ), f"Frame has incorrect dimensions: {frame.shape} != {media.camera.resolution}"

    media.close()


@pytest.mark.video
@pytest.mark.wireless
def test_get_frame_exists_webrtc() -> None:
    """Test that a frame can be retrieved from the WebRTC camera."""
    media = MediaManager(
        backend=MediaBackend.WEBRTC,
        signalling_host=SIGNALING_HOST,
    )
    time.sleep(2)
    frame = media.get_frame()
    if frame is None:
        print("Waiting extra time for WebRTC backend to get the first frame...")
        time.sleep(4)
        frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert (
        frame.shape[0] == media.camera.resolution[1]
        and frame.shape[1] == media.camera.resolution[0]
    ), f"Frame has incorrect dimensions: {frame.shape} != {media.camera.resolution}"

    media.close()


@pytest.mark.video
def test_get_frame_exists_all_resolutions(ipc_video_source: CameraSpecs) -> None:
    """Test that a frame can be retrieved at the default resolution.

    Note: changing resolution on the IPC reader does not change the
    daemon-side output resolution, so we only test that the default
    resolution works end-to-end.  Resolution validation logic is covered
    by ``test_change_resolution_errors``.
    """
    media = MediaManager(
        backend=MediaBackend.LOCAL,
        camera_specs=ipc_video_source,
    )
    time.sleep(1)
    frame = media.get_frame()
    resolution = ipc_video_source.default_resolution
    assert frame is not None, (
        f"No frame was retrieved from the camera at resolution {resolution}."
    )
    assert isinstance(frame, np.ndarray), (
        f"Frame is not a numpy array at resolution {resolution}."
    )
    assert frame.size > 0, f"Frame is empty at resolution {resolution}."
    assert (
        frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0]
    ), f"Frame has incorrect dimensions at resolution {resolution}: {frame.shape}"

    media.close()


@pytest.mark.video
@pytest.mark.parametrize("backend", VIDEO_BACKENDS)
def test_change_resolution_errors(backend: MediaBackend) -> None:
    """Test that changing resolution raises errors for invalid specs/resolutions."""
    media = MediaManager(
        backend=backend,
        signalling_host=SIGNALING_HOST
        if backend == MediaBackend.WEBRTC
        else "localhost",
    )
    media.camera.camera_specs = None
    with pytest.raises(RuntimeError):
        media.camera.set_resolution(CameraResolution.R1280x720at30fps)

    media.camera.camera_specs = MujocoCameraSpecs()
    with pytest.raises(RuntimeError):
        media.camera.set_resolution(CameraResolution.R1280x720at30fps)
    media.camera.camera_specs = ReachyMiniLiteCamSpecs()
    with pytest.raises(ValueError):
        media.camera.set_resolution(CameraResolution.R1280x720at30fps)

    media.close()
