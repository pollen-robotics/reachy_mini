import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import mediapipe as mp

from stewart_little_control.client import Client


class PoseEstimator:
    def predict(self, face_landmarks, image, max_x=0.1, max_y=0.1):
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
        return [
            int(face_landmark.x * img_w),
            int(face_landmark.y * img_h),
            face_landmark.z,
        ]


def main(draw=True):
    cap = cv.VideoCapture(0)
    client = Client()

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=1,
    )
    pose_estimator = PoseEstimator()

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
            client.send_pose(pose, offset_zero=True)

        cv.imshow("test_window", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
