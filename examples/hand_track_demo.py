"""Hand tracking demo for Reachy Mini."""

import time

import cv2
import numpy as np
from hand_tracker import HandTracker
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.io.cam_utils import find_camera


def draw_debug(img, palm_center):
    """Draw debug information on the image."""
    h, w, _ = img.shape
    draw_palm = [(-palm_center[0] + 1) / 2, (palm_center[1] + 1) / 2]  # [0, 1]
    cv2.circle(
        img,
        (int(w - draw_palm[0] * w), int(draw_palm[1] * h)),
        radius=5,
        color=(0, 0, 255),
        thickness=-1,
    )

    _target = [0.5, 0.5]
    cv2.circle(
        img,
        (int(_target[0] * w), int(_target[1] * h)),
        radius=5,
        color=(255, 0, 0),
        thickness=-1,
    )

    cv2.line(
        img,
        (int(draw_palm[0] * w), int(draw_palm[1] * h)),
        (int(_target[0] * w), int(_target[1] * h)),
        color=(0, 255, 0),
        thickness=2,
    )


cap = find_camera()

hand_tracker = HandTracker()
pose = np.eye(4)
euler_rot = np.array([0.0, 0.0, 0.0])
kp = 0.2
t0 = time.time()
with ReachyMini() as reachy_mini:
    try:
        while True:
            t = time.time() - t0

            success, img = cap.read()
            hands = hand_tracker.get_hands_positions(img)
            if hands is None:
                continue
            palm_center = hands[0]
            if palm_center is not None:
                palm_center[0] = -palm_center[0]  # Flip x-axis
                draw_debug(img, palm_center)

                target = [0, 0]
                error = np.array(target) - palm_center  # [-1, 1] [-1, 1]
                euler_rot += np.array([0.0, -kp * 0.1 * error[1], kp * error[0]])

                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                pose[:3, 3][2] = (
                    error[1] * 0.04
                )  # Adjust height based on vertical error
                pose[:3, 3][1] = (
                    error[0] * 0.02
                )  # Adjust height based on vertical error

                reachy_mini.set_target(head=pose)
            cv2.imshow("test_window", img)

            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
