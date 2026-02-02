import pytest


from reachy_mini.reachy_mini import ReachyMini
import time
import numpy as np
from reachy_mini.media.camera_constants import CameraResolution

@pytest.mark.wireless
def test_daemon_wireless_client_disconnection() -> None:
    with ReachyMini(media_backend="no_media", connection_mode="network") as mini:
        status = mini.client.get_status()
        assert status['state'] == "running"
        assert status['wireless_version'] is True
        assert not status['simulation_enabled']
        assert status['error'] is None
        assert status['backend_status']['motor_control_mode'] == "enabled"
        assert status['backend_status']['error'] is None
        assert isinstance(status['wlan_ip'], str)
        assert status['wlan_ip'].count('.') == 3
        assert all(0 <= int(part) <= 255 for part in status['wlan_ip'].split('.') if part.isdigit())

@pytest.mark.wireless_gstreamer
def test_daemon_wireless_gstreamer() -> None:
    with ReachyMini(media_backend="gstreamer") as mini:
        time.sleep(3)  # Give some time for the camera to initialize
        frame = mini.media.get_frame()
        assert frame is not None, "No frame was retrieved from the camera."
        assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
        assert frame.shape[0] == CameraResolution.R1280x720.value[1] and frame.shape[1] == CameraResolution.R1280x720.value[0], f"Frame has incorrect dimensions: {frame.shape}"
