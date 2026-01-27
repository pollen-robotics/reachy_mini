from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs, CameraResolution, MujocoCameraSpecs
from reachy_mini.media.media_manager import MediaManager, MediaBackend
import numpy as np
import pytest
import time
# import tempfile
import importlib.util

SIGNALING_HOST = "reachy-mini.local"

# Check if OpenCV is installed
_opencv_available = importlib.util.find_spec("cv2") is not None

#All video backends to test
VIDEO_BACKENDS = [
    pytest.param(backend, marks=pytest.mark.wireless) if backend == MediaBackend.WEBRTC else pytest.param(backend)
    for backend in MediaBackend
    if "NO_VIDEO" not in backend.name and backend != MediaBackend.NO_MEDIA
    and (_opencv_available or "OPENCV" not in backend.name)
]

@pytest.mark.video
@pytest.mark.parametrize("backend", VIDEO_BACKENDS)
def test_get_frame_exists(backend: MediaBackend) -> None:
    """Test that a frame can be retrieved from the camera and is not None."""
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
    time.sleep(2)
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert frame.shape[0] == media.camera.resolution[1] and frame.shape[1] == media.camera.resolution[0], f"Frame has incorrect dimensions: {frame.shape} != {media.camera.resolution}"

    # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
    #    cv2.imwrite(tmp_file.name, frame)
    #    print(f"Frame saved for inspection: {tmp_file.name}")

    media.close()

@pytest.mark.video
@pytest.mark.parametrize("backend", VIDEO_BACKENDS)
def test_get_frame_exists_all_resolutions(backend: MediaBackend) -> None:
    """Test that a frame can be retrieved from the camera for all supported resolutions."""
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
    for resolution in media.camera.camera_specs.available_resolutions:
        print(backend)
        print(media.camera.camera_specs.name)
        print(resolution)
        media.camera.close()
        media.camera.set_resolution(resolution)
        media.camera.open()
        time.sleep(2)
        frame = media.get_frame()
        assert frame is not None, f"No frame was retrieved from the camera at resolution {resolution}."
        assert isinstance(frame, np.ndarray), f"Frame is not a numpy array at resolution {resolution}."
        assert frame.size > 0, f"Frame is empty at resolution {resolution}."
        assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions at resolution {resolution}: {frame.shape}"

    media.close()

@pytest.mark.video
@pytest.mark.parametrize("backend", VIDEO_BACKENDS)
def test_change_resolution_errors(backend: MediaBackend) -> None:
    """Test that changing resolution raises a runtime error if not allowed."""
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
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
