from hand_tracker import HandTracker
import cv2
import time

from stewart_little_control import Client
import numpy as np
from scipy.spatial.transform import Rotation as R

from noise import pnoise1


def smooth_movement(t, speed=0.5, scale=0.8):
    return pnoise1(t * speed) * scale

def draw_debug(img, palm_center):
    h, w, _ = img.shape
    draw_palm = [(-palm_center[0]+1)/2, (palm_center[1]+1)/2] # [0, 1]
    cv2.circle(
        img,
        (int(w-draw_palm[0] * w), int(draw_palm[1] * h)),
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



cap = cv2.VideoCapture(4)
# cap = cv2.VideoCapture(0)

client = Client()
hand_tracker = HandTracker()
pose = np.eye(4)
pose[:3, 3][2] = 0.177  # Set the height of the head
euler_rot = np.array([0.0, 0.0, 0.0])
kp = 0.3
t0 = time.time()
while True:
    t = time.time() - t0
    left_antenna = smooth_movement(t)
    right_antenna = smooth_movement(t + 200)

    success, img = cap.read()

    palm_center = hand_tracker.get_hand_position(img)
    if palm_center is not None:
        palm_center[0] = -palm_center[0]  # Flip x-axis
        print(palm_center)
        draw_debug(img, palm_center)

        target = [0, 0]
        error = np.array(target) - palm_center  # [-1, 1] [-1, 1]
        euler_rot += np.array([0.0, -kp * 0.1 * error[1], kp * error[0]])

        rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        pose[:3, :3] = rot_mat
        pose[:3, 3][2] = (
            error[1] * 0.04 + 0.177
        )  # Adjust height based on vertical error

        # antennas = [left_antenna, right_antenna]
        antennas = [0, 0]
        client.send_pose(pose, antennas=antennas)
    cv2.imshow("test_window", img)

    cv2.waitKey(1)
