"""Head tracking demo for Reachy Mini."""

import time

import cv2
import numpy as np
import rerun as rr
from head_tracker import HeadTracker
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.io.cam_utils import find_camera


def draw_rr_debug(img, eye_center, roll):
    """Draw debug information on the image for Rerun."""
    _eye_center = eye_center.copy() + 0.5  # [0, 1]
    h, w, _ = img.shape
    rr.log(
        "face_tracker/eye_center",
        rr.Points2D(
            [[(_eye_center[0]) * w, (_eye_center[1]) * h], [0.5 * w, 0.5 * h]],
            radii=rr.Radius.ui_points([10]),
            colors=[[0, 0, 255], [255, 0, 0]],
            # labels=[["eye_center"], ["target"]],
        ),
    )
    rr.log(
        "face_tracker/lines",
        rr.LineStrips2D(
            [
                [
                    [(_eye_center[0]) * w, (_eye_center[1]) * h],
                    [
                        _eye_center[0] * w + 100 * np.cos(roll),
                        _eye_center[1] * h + 100 * np.sin(roll),
                    ],
                ],
                [[(_eye_center[0]) * w, (_eye_center[1]) * h], [0.5 * w, 0.5 * h]],
            ],
            colors=[[255, 255, 0], [0, 255, 0]],
            # labels=["roll", "error"],
            radii=rr.Radius.ui_points([1, 1]),
        ),
    )


cap = find_camera()

head_tracker = HeadTracker()
pose = np.eye(4)
euler_rot = np.array([0.0, 0.0, 0.0])
kp = 0.3
t0 = time.time()
with ReachyMini() as reachy_mini:
    app_id, recording_id = reachy_mini.get_rerun_ids()
    print(f"Application ID: {app_id}, Recording ID: {recording_id}")

    rr.init(
        application_id=app_id,
        recording_id=recording_id,
        spawn=True,
    )

    try:
        while True:
            t = time.time() - t0
            rr.set_time("deamon", timestamp=time.time())
            success, img = cap.read()

            ret, encoded_image = cv2.imencode(".jpg", img)

            rr.log(
                "face_tracker/image",
                rr.EncodedImage(contents=encoded_image, media_type="image/jpeg"),
            )

            eye_center, roll = head_tracker.get_head_position(img)
            if eye_center is not None:
                draw_rr_debug(img, eye_center, roll)

                target = [0, 0]
                error = np.array(target) - eye_center  # [-1, 1] [-1, 1]
                euler_rot += np.array(
                    [kp * roll * 0.1, -kp * 0.1 * error[1], kp * error[0]]
                )

                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                pose[:3, 3][2] = (
                    error[1] * 0.04
                )  # Adjust height based on vertical error

                reachy_mini.set_target(head=pose)

    except KeyboardInterrupt:
        pass
