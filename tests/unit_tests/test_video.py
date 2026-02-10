from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs, CameraResolution, MujocoCameraSpecs
from reachy_mini.media.media_manager import MediaManager, MediaBackend
import numpy as np
import pytest
import time
# import tempfile
# import cv2


@pytest.mark.video
def test_get_frame_exists() -> None:
    """Test that a frame can be retrieved from the camera and is not None."""
    media = MediaManager(backend=MediaBackend.DEFAULT)
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert frame.shape[0] == media.camera.resolution[1] and frame.shape[1] == media.camera.resolution[0], f"Frame has incorrect dimensions: {frame.shape}"

    # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
    #    cv2.imwrite(tmp_file.name, frame)
    #    print(f"Frame saved for inspection: {tmp_file.name}")    

@pytest.mark.video
def test_get_frame_exists_all_resolutions() -> None:
    """Test that a frame can be retrieved from the camera for all supported resolutions."""
    media = MediaManager(backend=MediaBackend.DEFAULT)
    for resolution in media.camera.camera_specs.available_resolutions:
        media.camera.set_resolution(resolution)
        frame = media.get_frame()
        assert frame is not None, f"No frame was retrieved from the camera at resolution {resolution}."
        assert isinstance(frame, np.ndarray), f"Frame is not a numpy array at resolution {resolution}."
        assert frame.size > 0, f"Frame is empty at resolution {resolution}."
        assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions at resolution {resolution}: {frame.shape}" 

@pytest.mark.video
def test_change_resolution_errors() -> None:
    """Test that changing resolution raises a runtime error if not allowed."""
    media = MediaManager(backend=MediaBackend.DEFAULT)
    media.camera.camera_specs = None
    with pytest.raises(RuntimeError):
        media.camera.set_resolution(CameraResolution.R1280x720)

    media.camera.camera_specs = MujocoCameraSpecs()
    with pytest.raises(RuntimeError):
        media.camera.set_resolution(CameraResolution.R1280x720)
    media.camera.camera_specs = ReachyMiniLiteCamSpecs()
    with pytest.raises(ValueError):
        media.camera.set_resolution(CameraResolution.R1280x720)


@pytest.mark.video_gstreamer
def test_get_frame_exists_gstreamer() -> None:
    """Test that a frame can be retrieved from the camera and is not None."""
    media = MediaManager(backend=MediaBackend.GSTREAMER)
    time.sleep(2)  # Give some time for the camera to initialize
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert frame.shape[0] == media.camera.resolution[1] and frame.shape[1] == media.camera.resolution[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video_gstreamer
def test_get_frame_exists_all_resolutions_gstreamer() -> None:
    """Test that a frame can be retrieved from the camera for all supported resolutions."""
    media = MediaManager(backend=MediaBackend.GSTREAMER)
    time.sleep(2)  # Give some time for the camera to initialize

    for resolution in media.camera.camera_specs.available_resolutions:
        media.camera.close()
        media.camera.set_resolution(resolution)
        media.camera.open()
        time.sleep(2)  # Give some time for the camera to adjust to new resolution
        frame = media.get_frame()
        assert frame is not None, f"No frame was retrieved from the camera at resolution {resolution}."
        assert isinstance(frame, np.ndarray), f"Frame is not a numpy array at resolution {resolution}."
        assert frame.size > 0, f"Frame is empty at resolution {resolution}."
        assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions at resolution {resolution}: {frame.shape}"

@pytest.mark.video_gstreamer
def test_change_resolution_errors_gstreamer() -> None:
    """Test that changing resolution raises a runtime error if not allowed."""
    media = MediaManager(backend=MediaBackend.GSTREAMER)
    time.sleep(1)  # Give some time for the camera to initialize
    with pytest.raises(RuntimeError):
        media.camera.set_resolution(media.camera.camera_specs.available_resolutions[0])
