"""Demonstrate how to make Reachy Mini look at a point in an image.

When you click on the image, Reachy Mini will look at the point you clicked on.
It uses OpenCV to capture video from a camera and display it, and Reachy Mini's
look_at_image method to make the robot look at the specified point.

Note: The daemon must be running before executing this script.
"""

import cv2

from reachy_mini import ReachyMini
from reachy_mini.utils.camera import find_camera

click_x, click_y = 0, 0
just_clicked = False


def click(event, x, y, flags, param):
    """Handle mouse click events to get the coordinates of the click."""
    global click_x, click_y, just_clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        just_clicked = True
        click_x, click_y = x, y


def main():
    """Show the camera feed from Reachy Mini and make it look at clicked points."""
    global click_x, click_y, just_clicked

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
        cap = cv2.VideoCapture(args.camera_opencv_id)
    else:
        cap = find_camera()
        if cap is None:
            print(
                "Could not automatically find a camera, please specify the camera index with --camera-opencv-id"
            )
            return

    if not cap.isOpened():
        print(f"Failed to open camera with index {args.camera_opencv_id}")
        return

    cv2.namedWindow("Reachy Mini Camera")
    cv2.setMouseCallback("Reachy Mini Camera", click)

    print("Click on the image to make ReachyMini look at that point.")
    print("Press 'q' to quit the camera feed.")
    with ReachyMini() as reachy_mini:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                continue

            cv2.imshow("Reachy Mini Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break

            if just_clicked:
                reachy_mini.look_at_image(click_x, click_y, duration=0.3)
                just_clicked = False


if __name__ == "__main__":
    main()
