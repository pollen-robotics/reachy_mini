import platform

import cv2
from cv2_enumerate_cameras import enumerate_cameras


def find_camera(
    vid: int = 0x0C45,
    pid: int = 0x636D,
    apiPreference: int = cv2.CAP_ANY,
) -> cv2.VideoCapture | None:
    if platform.system() == 'Linux':
        apiPreference = cv2.CAP_V4L2        

    for c in enumerate_cameras(apiPreference):
        if c.vid == vid and c.pid == pid:
            return cv2.VideoCapture(c.index, c.backend)
    return None


if __name__ == "__main__":
    cam = find_camera()

    if cam is None:
        print("Camera not found")
    else:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
