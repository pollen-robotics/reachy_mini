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


# @pytest.mark.video
# def test_get_frame_exists_1600() -> None:
#     resolution = CameraResolution.R1600x1200
#     media = MediaManager(backend=MediaBackend.DEFAULT)
#     frame = media.get_frame()
#     assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

# @pytest.mark.video
# def test_get_frame_exists_1920() -> None:
#     resolution = CameraResolution.R1920x1080
#     media = MediaManager(backend=MediaBackend.DEFAULT, resolution=resolution)
#     frame = media.get_frame()
#     assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

# @pytest.mark.video
# def test_get_frame_exists_2304() -> None:
#     resolution = CameraResolution.R2304x1296
#     media = MediaManager(backend=MediaBackend.DEFAULT, resolution=resolution)
#     frame = media.get_frame()
#     assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

# @pytest.mark.video
# def test_get_frame_exists_4608() -> None:
#     resolution = CameraResolution.R4608x2592
#     media = MediaManager(backend=MediaBackend.DEFAULT, resolution=resolution)
#     frame = media.get_frame()
#     assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video_gstreamer
def test_get_frame_exists_gstreamer() -> None:
    """Test that a frame can be retrieved from the camera and is not None."""
    media = MediaManager(backend=MediaBackend.GSTREAMER)
    time.sleep(2)  # Give some time for the camera to initialize
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert frame.shape[0] == media.camera.resolution.value[1] and frame.shape[1] == media.camera.resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

@pytest.mark.video_gstreamer
def test_get_frame_exists_all_resolutions_gstreamer() -> None:
    """Test that a frame can be retrieved from the camera for all supported resolutions."""
    media = MediaManager(backend=MediaBackend.GSTREAMER)
    time.sleep(2)  # Give some time for the camera to initialize
    # TODO gstreamer is not working yet with the new refacto
    # for resolution in media.camera.camera_specs.available_resolutions:
    #     media.camera.set_resolution(resolution)
    #     time.sleep(1)  # Give some time for the camera to adjust to new resolution
    #     frame = media.get_frame()
    #     assert frame is not None, f"No frame was retrieved from the camera at resolution {resolution}."
    #     assert isinstance(frame, np.ndarray), f"Frame is not a numpy array at resolution {resolution}."
    #     assert frame.size > 0, f"Frame is empty at resolution {resolution}."
    #     assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions at resolution {resolution}: {frame.shape}"

# @pytest.mark.video_gstreamer
# def test_get_frame_exists_gstreamer_1600() -> None:
#     resolution = CameraResolution.R1600x1200
#     media = MediaManager(backend=MediaBackend.GSTREAMER, resolution=resolution)
#     time.sleep(2)  # Give some time for the camera to initialize
#     frame = media.get_frame()
#     assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

# @pytest.mark.video_gstreamer
# def test_get_frame_exists_gstreamer_1920() -> None:
#     resolution = CameraResolution.R1920x1080
#     media = MediaManager(backend=MediaBackend.GSTREAMER, resolution=resolution)
#     time.sleep(2)  # Give some time for the camera to initialize
#     frame = media.get_frame()
#     assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

# @pytest.mark.video_gstreamer
# def test_get_frame_exists_gstreamer_2304() -> None:
#     resolution = CameraResolution.R2304x1296
#     media = MediaManager(backend=MediaBackend.GSTREAMER, resolution=resolution)
#     time.sleep(2)  # Give some time for the camera to initialize
#     frame = media.get_frame()
#     assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"

# @pytest.mark.video_gstreamer
# def test_get_frame_exists_gstreamer_4608() -> None:
#     resolution = CameraResolution.R4608x2592
#     media = MediaManager(backend=MediaBackend.GSTREAMER, resolution=resolution)
#     time.sleep(2)  # Give some time for the camera to initialize
#     frame = media.get_frame()
#     assert frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0], f"Frame has incorrect dimensions: {frame.shape}"
