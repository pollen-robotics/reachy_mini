"""Camera utility for Reachy Mini."""

import platform

import cv2
from cv2_enumerate_cameras import enumerate_cameras


def find_camera(
    vid: int = 0x0C45,
    pid: int = 0x636D,
    apiPreference: int = cv2.CAP_ANY,
) -> cv2.VideoCapture | None:
    """Find and return a camera with the specified VID and PID.

    Args:
        vid (int): Vendor ID of the camera. Default is 0x0C45 (Arducam).
        pid (int): Product ID of the camera. Default is 0x636D (Arducam).
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


if __name__ == "__main__":
    from reachy_mini.media.camera_constants import CameraResolution

    cam = find_camera()

    if cam is None:
        print("Camera not found")
    else:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CameraResolution.R1280x720.value[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraResolution.R1280x720.value[1])

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
