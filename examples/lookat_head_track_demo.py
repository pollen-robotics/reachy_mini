"""Reachy Mini Look At Head Tracking Example."""

import cv2
from head_tracker import HeadTracker

from reachy_mini import ReachyMini
from reachy_mini.io.cam_utils import find_camera

cap = find_camera()
head_tracker = HeadTracker()

with ReachyMini() as reachy_mini:
    try:
        while True:
            success, img = cap.read()
            if success:
                eye_center, roll = head_tracker.get_head_position(img)
                if eye_center is not None:
                    h, w, _ = img.shape
                    eye_center = (eye_center + 1) / 2
                    eye_center[0] *= w
                    eye_center[1] *= h
                    cv2.circle(
                        img,
                        center=(int(eye_center[0]), int(eye_center[1])),
                        radius=5,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    reachy_mini.look_at_image(*eye_center, duration=0.0)

            cv2.imshow("image", img)

            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
