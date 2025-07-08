"""Show the camera feed from Reachy Mini using OpenCV.

This script tries to automatically find the Reachy Mini camera (by product ID), or you can specify the OpenCV camera index with --camera-opencv-id.

Requirements:
    - Python 3.8+
    - OpenCV (cv2)
    - reachy_mini package

Usage:
    python show_cam.py [--camera-opencv-id <index>]

- If no camera index is provided, the script will attempt to auto-detect the camera.
- Press 'q' in the window to quit.
"""

import cv2

from reachy_mini.io.cam_utils import find_camera


def main():
    """Show the camera feed from Reachy Mini."""
    import argparse

    parser = argparse.ArgumentParser(description="Show camera feed from Reachy Mini.")
    parser.add_argument(
        "--camera-opencv-id",
        type=int,
        default=None,
        help="Camera index to use (if not specified, it will automatically tries to find it).",
    )
    args = parser.parse_args()

    if args.camera_opencv_id is not None:
        cam = cv2.VideoCapture(args.camera_opencv_id)
    else:
        cam = find_camera()
        if cam is None:
            print(
                "Could not automatically find a camera, please specify the camera index with --camera-opencv-id"
            )
            return

    if not cam.isOpened():
        print(f"Failed to open camera with index {args.camera_opencv_id}")
        return

    print("Press 'q' to quit the camera feed.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Reachy Mini Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
