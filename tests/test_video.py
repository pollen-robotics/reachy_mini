from reachy_mini.media.media_manager import MediaManager, MediaBackend
import numpy as np
import pytest
import time
# import tempfile
# import tempfile
# import cv2


@pytest.mark.video
def test_get_frame_exists():
    """Test that a frame can be retrieved from the camera and is not None."""
    media = MediaManager(backend=MediaBackend.DEFAULT)
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."

    # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
    #    cv2.imwrite(tmp_file.name, frame)
    #    print(f"Frame saved for inspection: {tmp_file.name}")

@pytest.mark.video_gstreamer
def test_get_frame_exists_gstreamer():
    """Test that a frame can be retrieved from the camera and is not None."""
    media = MediaManager(backend=MediaBackend.GSTREAMER)
    time.sleep(2)  # Give some time for the camera to initialize
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, bytes), "Frame is not a bytes object."
    assert len(frame) > 0, "Frame is empty."
    # with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
    #    tmp_file.write(frame)
    #    print(f"Frame saved for inspection: {tmp_file.name}")