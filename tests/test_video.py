from reachy_mini.media.camera_constants import CameraResolution
from reachy_mini.media.media_manager import MediaManager, MediaBackend
import numpy as np
import pytest
import time
# import tempfile
# import cv2


@pytest.mark.video
def test_get_frame_exists() -> None:
    """Test that a frame can be retrieved from the camera and is not None."""
    resolution = CameraResolution.R1280x720
    media = MediaManager(backend=MediaBackend.DEFAULT, resolution=resolution)
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

    # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
    #    cv2.imwrite(tmp_file.name, frame)
    #    print(f"Frame saved for inspection: {tmp_file.name}")    

@pytest.mark.video
def test_get_frame_exists_1600() -> None:
    resolution = CameraResolution.R1600x1200
    media = MediaManager(backend=MediaBackend.DEFAULT, resolution=resolution)
    frame = media.get_frame()
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video
def test_get_frame_exists_1920() -> None:
    resolution = CameraResolution.R1920x1080
    media = MediaManager(backend=MediaBackend.DEFAULT, resolution=resolution)
    frame = media.get_frame()
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video
def test_get_frame_exists_2304() -> None:
    resolution = CameraResolution.R2304x1296
    media = MediaManager(backend=MediaBackend.DEFAULT, resolution=resolution)
    frame = media.get_frame()
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video
def test_get_frame_exists_4608() -> None:
    resolution = CameraResolution.R4608x2592
    media = MediaManager(backend=MediaBackend.DEFAULT, resolution=resolution)
    frame = media.get_frame()
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video_gstreamer
def test_get_frame_exists_gstreamer() -> None:
    """Test that a frame can be retrieved from the camera and is not None."""
    resolution = CameraResolution.R1280x720
    media = MediaManager(backend=MediaBackend.GSTREAMER, resolution=resolution)
    time.sleep(2)  # Give some time for the camera to initialize
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video_gstreamer
def test_get_frame_exists_gstreamer_1600() -> None:
    resolution = CameraResolution.R1600x1200
    media = MediaManager(backend=MediaBackend.GSTREAMER, resolution=resolution)
    time.sleep(2)  # Give some time for the camera to initialize
    frame = media.get_frame()
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video_gstreamer
def test_get_frame_exists_gstreamer_1920() -> None:
    resolution = CameraResolution.R1920x1080
    media = MediaManager(backend=MediaBackend.GSTREAMER, resolution=resolution)
    time.sleep(2)  # Give some time for the camera to initialize
    frame = media.get_frame()
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video_gstreamer
def test_get_frame_exists_gstreamer_2304() -> None:
    resolution = CameraResolution.R2304x1296
    media = MediaManager(backend=MediaBackend.GSTREAMER, resolution=resolution)
    time.sleep(2)  # Give some time for the camera to initialize
    frame = media.get_frame()
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video_gstreamer
def test_get_frame_exists_gstreamer_4608() -> None:
    resolution = CameraResolution.R4608x2592
    media = MediaManager(backend=MediaBackend.GSTREAMER, resolution=resolution)
    time.sleep(2)  # Give some time for the camera to initialize
    frame = media.get_frame()
    assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"
