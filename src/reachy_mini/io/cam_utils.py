import platform

import cv2 as cv
from cv2_enumerate_cameras import enumerate_cameras
from cv2_enumerate_cameras.camera_info import CameraInfo


def find_camera_opencv_id(vid="0C45", pid="636D") -> list[int]:
    vid = int(vid, base=16)
    pid = int(pid, base=16)

    if platform.system() == "Linux":
        cameras = enumerate_cameras(cv.CAP_V4L2)

    elif platform.system() == "Windows":
        cameras = enumerate_cameras(cv.CAP_DSHOW)

    elif platform.system() == "Darwin":
        cameras = _macos_enumerate_cameras()

    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")

    cameras = [
        cam_info.index
        for cam_info in cameras
        if cam_info.vid == vid and cam_info.pid == pid
    ]
    return cameras


def _macos_enumerate_cameras() -> list[CameraInfo]:
    import re

    import AVFoundation

    _VID_RE = re.compile(r"VendorID_(\d+)")
    _PID_RE = re.compile(r"ProductID_(\d+)")

    cams: list[CameraInfo] = []

    devs = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
        AVFoundation.AVMediaTypeVideo
    )
    devs = sorted(devs, key=lambda d: str(d.uniqueID()))

    for idx, d in enumerate(devs):
        model = str(d.modelID())
        vid_m = _VID_RE.search(model)
        pid_m = _PID_RE.search(model)
        cams.append(
            CameraInfo(
                index=idx,
                name=str(d.localizedName()),
                path=None,  # macOS does not provide a path
                vid=int(vid_m.group(1)) if vid_m else None,
                pid=int(pid_m.group(1)) if pid_m else None,
                backend=cv.CAP_AVFOUNDATION,
            )
        )
    return cams


def main():
    ids = find_camera_opencv_id()

    if not ids:
        print("No camera found with the specified VID and PID.")
        return

    if len(ids) > 1:
        print(f"Multiple cameras found: {ids}. Using the first one.")

    print(f"Using camera with OpenCV ID: {ids[0]}")
    cam_id = ids[0]

    cap = cv.VideoCapture(cam_id)

    while True:
        b, img = cap.read()
        if b:
            cv.imshow("cam", img)
            cv.waitKey(33)


if __name__ == "__main__":
    main()
