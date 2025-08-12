"""Reachy Mini Face Tracking Example.

When running this example, the Reachy Mini will track your face and react to your head movements.
It will look at the center of your face and adjust its head and antennas accordingly.

This example uses MediaPipe to detect face landmarks and estimate the head pose.
The estimated pose is then used to control the Reachy Mini's head and antennas.
"""

import cv2 as cv
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.camera import find_camera


class PoseEstimator:
    """Estimate the head pose from face landmarks detected by MediaPipe."""

    def predict(self, face_landmarks, image, max_x=0.1, max_y=0.1):
        """Estimate the head pose from face landmarks.

        Args:
            face_landmarks: The face landmarks detected by MediaPipe.
            image: The input image from which the landmarks were detected.
            max_x: Maximum x-coordinate for the head pose.
            max_y: Maximum y-coordinate for the head pose.

        Returns:
            pose: A 4x4 numpy array representing the head pose in the Reachy Mini
            coordinate system.

        """
        h, w, _ = image.shape

        left_eye = np.array(
            (face_landmarks.landmark[33].x, face_landmarks.landmark[33].y)
        )
        left_eye = left_eye * 2 - 1

        right_eye = np.array(
            (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y)
        )
        right_eye = right_eye * 2 - 1

        eye_center = np.mean([left_eye, right_eye], axis=0)

        x = eye_center[0] * max_x
        y = eye_center[1] * max_y
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

        pose = np.eye(4)

        rot = R.from_euler("xyz", [roll, 0, 0])
        pose[:3, :3] = rot.as_matrix()

        pose[1, 3] = x
        pose[2, 3] = -y

        return pose

    def get_landmark_coords(self, face_landmark, img_h, img_w):
        """Get the coordinates of a face landmark in pixel space."""
        return [
            int(face_landmark.x * img_w),
            int(face_landmark.y * img_h),
            face_landmark.z,
        ]


def main(draw=True):
    """Run the Reachy Mini face tracking example."""
    cap = find_camera()

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=1,
    )
    pose_estimator = PoseEstimator()

    with ReachyMini() as reachy_mini:
        try:
            while True:
                success, img = cap.read()

                if not success:
                    print("Failed to capture image")
                    continue

                results = face_mesh.process(img)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    if draw:
                        mp.solutions.drawing_utils.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
                        )
                    pose = pose_estimator.predict(face_landmarks, img)
                    reachy_mini.set_target(head=pose, antennas=[0, 0])

                cv.imshow("test_window", img)
                cv.waitKey(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
