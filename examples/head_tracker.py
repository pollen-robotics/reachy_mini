import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import mediapipe as mp


class HeadTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.05,
            min_tracking_confidence=0.5,
            max_num_faces=1,
        )

    def get_eyes(self, img):
        results = self.face_mesh.process(img)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            left_eye = np.array(
                (face_landmarks.landmark[33].x, face_landmarks.landmark[33].y)
            )
            left_eye = left_eye * 2 - 1

            right_eye = np.array(
                (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y)
            )
            right_eye = right_eye * 2 - 1
            return left_eye, right_eye

        return None, None

    def get_eyes_from_landmarks(self, face_landmarks):
        left_eye = np.array(
            (face_landmarks.landmark[33].x, face_landmarks.landmark[33].y)
        )
        left_eye = left_eye * 2 - 1

        right_eye = np.array(
            (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y)
        )
        right_eye = right_eye * 2 - 1
        return left_eye, right_eye

    def get_eye_center(self, face_landmarks):
        left_eye, right_eye = self.get_eyes_from_landmarks(face_landmarks)
        eye_center = np.mean([left_eye, right_eye], axis=0)
        return eye_center

    def get_roll(self, face_landmarks):
        left_eye = np.array(
            (face_landmarks.landmark[33].x, face_landmarks.landmark[33].y)
        )
        left_eye = left_eye * 2 - 1

        right_eye = np.array(
            (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y)
        )
        right_eye = right_eye * 2 - 1
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

        return roll

    def get_head_position(self, img):
        results = self.face_mesh.process(img)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            eye_center = self.get_eye_center(face_landmarks)  # [-1, 1] [-1, 1]
            roll = self.get_roll(face_landmarks)

            return eye_center, roll

        return None, None


if __name__ == "__main__":
    # cap = cv2.VideoCapture(4)
    cap = cv2.VideoCapture(0)
    head_tracker = HeadTracker()
    while True:
        success, img = cap.read()
        if not success:
            break

        eye_center, roll = head_tracker.get_head_position(img)
        if eye_center is not None:
            _eye_center = (eye_center.copy() + 1) / 2  # [0, 1]
            h, w, _ = img.shape
            cv2.circle(
                img,
                (int(_eye_center[0] * w), int(_eye_center[1] * h)),
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            )
        cv2.imshow("test_window", img)
        cv2.waitKey(1)
