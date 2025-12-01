"""Camera utility for Reachy Mini."""

import platform
from logging import Logger
from typing import Optional, Tuple, cast

import cv2
from cv2_enumerate_cameras import enumerate_cameras

from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraSpecs,
    OlderRPiCamSpecs,
    ReachyMiniLiteCamSpecs,
    ReachyMiniWirelessCamSpecs,
)

try:
    import gi
except ImportError as e:
    raise ImportError(
        "The 'gi' module is required for GStreamerCamera but could not be imported. \
                      Please install the GStreamer backend: pip install .[gstreamer]."
    ) from e

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")


from gi.repository import Gst  # noqa: E402


def find_camera(
    apiPreference: int = cv2.CAP_ANY, no_cap: bool = False
) -> Tuple[Optional[cv2.VideoCapture], Optional[CameraSpecs]]:
    """Find and return the Reachy Mini camera.

    Looks for the Reachy Mini camera first, then Arducam, then older Raspberry Pi Camera. Returns None if no camera is found.

    Args:
        apiPreference (int): Preferred API backend for the camera. Default is cv2.CAP_ANY.
        no_cap (bool): If True, close the camera after finding it. Default is False.

    Returns:
        cv2.VideoCapture | None: A VideoCapture object if the camera is found and opened successfully, otherwise None.

    """
    cap = find_camera_by_vid_pid(
        ReachyMiniLiteCamSpecs.vid, ReachyMiniLiteCamSpecs.pid, apiPreference
    )
    if cap is not None:
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # type: ignore
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if no_cap:
            cap.release()
        return cap, cast(CameraSpecs, ReachyMiniLiteCamSpecs)

    cap = find_camera_by_vid_pid(
        ReachyMiniLiteCamSpecs.vid, OlderRPiCamSpecs.pid, apiPreference
    )
    if cap is not None:
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # type: ignore
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if no_cap:
            cap.release()
        return cap, cast(CameraSpecs, OlderRPiCamSpecs)

    cap = find_camera_by_vid_pid(ArducamSpecs.vid, ArducamSpecs.pid, apiPreference)
    if cap is not None:
        if no_cap:
            cap.release()
        return cap, cast(CameraSpecs, ArducamSpecs)

    return None, None


def find_camera_by_vid_pid(
    vid: int = ReachyMiniLiteCamSpecs.vid,
    pid: int = ReachyMiniLiteCamSpecs.pid,
    apiPreference: int = cv2.CAP_ANY,
) -> cv2.VideoCapture | None:
    """Find and return a camera with the specified VID and PID.

    Args:
        vid (int): Vendor ID of the camera. Default is ReachyMiniCamera
        pid (int): Product ID of the camera. Default is ReachyMiniCamera
        apiPreference (int): Preferred API backend for the camera. Default is cv2.CAP_ANY.

    Returns:
        cv2.VideoCapture | None: A VideoCapture object if the camera is found and opened successfully, otherwise None.

    """
    if platform.system() == "Linux":
        apiPreference = cv2.CAP_V4L2

    selected_cap = None
    for c in enumerate_cameras(apiPreference):
        if c.vid == vid and c.pid == pid:
            # the Arducam camera creates two /dev/videoX devices
            # that enumerate_cameras cannot differentiate
            try:
                cap = cv2.VideoCapture(c.index, c.backend)
                if cap.isOpened():
                    selected_cap = cap
            except Exception as e:
                print(f"Error opening camera {c.index}: {e}")
    return selected_cap


def get_video_device(logger: Logger) -> Tuple[str | Optional[CameraSpecs]]:
    """Use Gst.DeviceMonitor to find the unix camera path /dev/videoX.

    Returns the device path (e.g., '/dev/video2'), or '' if not found.
    """
    monitor = Gst.DeviceMonitor()
    monitor.add_filter("Video/Source")
    monitor.start()

    cam_names = ["Reachy", "Arducam_12MP", "imx708"]

    devices = monitor.get_devices()
    for cam_name in cam_names:
        for device in devices:
            name = device.get_display_name()
            device_props = device.get_properties()

            if cam_name in name:
                if device_props and device_props.has_field("api.v4l2.path"):
                    device_path = device_props.get_string("api.v4l2.path")
                    camera_specs = (
                        cast(CameraSpecs, ArducamSpecs)
                        if cam_name == "Arducam_12MP"
                        else cast(CameraSpecs, ReachyMiniLiteCamSpecs)
                    )
                    logger.debug(f"Found {cam_name} camera at {device_path}")
                    monitor.stop()
                    return str(device_path), camera_specs
                elif cam_name == "imx708":
                    camera_specs = cast(CameraSpecs, ReachyMiniWirelessCamSpecs)
                    logger.debug(f"Found {cam_name} camera")
                    monitor.stop()
                    return cam_name, camera_specs
    monitor.stop()
    logger.warning("No camera found.")
    return "", None


if __name__ == "__main__":
    from reachy_mini.media.camera_constants import ArduCamResolution

    cam, _ = find_camera()
    if cam is None:
        exit("Camera not found")

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, ArduCamResolution.R1280x720.value[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, ArduCamResolution.R1280x720.value[1])

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
