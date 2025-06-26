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


def _macos_enumerate_cameras():
    import re
    import subprocess

    CAMERA_HEADER_RE = re.compile(r"\n\s*\n(?=\s{4}\S)")  # same splitter as before
    NAME_RE = re.compile(r"^\s{4}(.+?):", re.MULTILINE)  # 4-space indent + colon
    VENDOR_RE = re.compile(r"VendorID_(\d+)\b")
    PRODUCT_RE = re.compile(r"ProductID_(\d+)\b")

    text = subprocess.check_output(
        ["system_profiler", "-detailLevel", "mini", "SPCameraDataType"],
        text=True,
        timeout=2,
    )

    blocks = CAMERA_HEADER_RE.split(text.strip())[1:]  # drop the "Camera:" header
    cameras = []

    for index, blk in enumerate(blocks):
        name = NAME_RE.search(blk)
        vendor = VENDOR_RE.search(blk)
        prod = PRODUCT_RE.search(blk)

        name = name.group(1) if name else ""
        vendor = int(vendor.group(1)) if vendor else None
        prod = int(prod.group(1)) if prod else None

        cameras.append(
            CameraInfo(
                index=index,
                name=name,
                path=None,
                vid=vendor,
                pid=prod,
                backend=cv.CAP_AVFOUNDATION,
            )
        )

    return cameras


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
