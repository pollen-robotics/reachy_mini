"""Demonstrate how to make Reachy Mini look at a point in an image.

When you click on the image, Reachy Mini will look at the point you clicked on.
It uses OpenCV to capture video from a camera and display it, and Reachy Mini's
look_at_image method to make the robot look at the specified point.

Note: The daemon must be running before executing this script.
"""

import cv2

from reachy_mini import ReachyMini


def click(event, x, y, flags, param):
    """Handle mouse click events to get the coordinates of the click."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param["just_clicked"] = True
        param["x"] = x
        param["y"] = y


def main():
    """Show the camera feed from Reachy Mini and make it look at clicked points."""
    state = {"x": 0, "y": 0, "just_clicked": False}

    cv2.namedWindow("Reachy Mini Camera")
    cv2.setMouseCallback("Reachy Mini Camera", click, param=state)

    print("Click on the image to make ReachyMini look at that point.")
    print("Press 'q' to quit the camera feed.")
    with ReachyMini(use_sim=False) as reachy_mini:
        while True:
            frame = reachy_mini.media.get_frame()
            if frame is None:
                print("Failed to grab frame.")
                continue

            cv2.imshow("Reachy Mini Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break

            if state["just_clicked"]:
                reachy_mini.look_at_image(state["x"], state["y"], duration=0.3)
                state["just_clicked"] = False


if __name__ == "__main__":
    main()
