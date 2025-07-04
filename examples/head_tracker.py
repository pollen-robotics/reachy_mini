"""Head Tracker using MediaPipe Face Mesh to estimate head pose."""

try:
    import mediapipe as mp
except ImportError:
    print("mediapipe is not installed. Install it using 'pip install mediapipe'.")
    exit()

import numpy as np


class HeadTracker:
    """Head Tracker using MediaPipe Face Mesh to estimate head pose."""

    def __init__(self):
        """Initialize the Head Tracker."""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.05,
            min_tracking_confidence=0.5,
            max_num_faces=1,
        )

    def get_eyes(self, img):
        """Get the coordinates of the eyes from the image."""
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
        """Get the coordinates of the eyes from face landmarks."""
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
        """Get the center of the eyes from face landmarks."""
        left_eye, right_eye = self.get_eyes_from_landmarks(face_landmarks)
        eye_center = np.mean([left_eye, right_eye], axis=0)
        return eye_center

    def get_roll(self, face_landmarks):
        """Calculate the roll of the head based on eye positions."""
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
        """Get the head position and roll from the image."""
        results = self.face_mesh.process(img)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            eye_center = self.get_eye_center(face_landmarks)  # [-1, 1] [-1, 1]
            roll = self.get_roll(face_landmarks)

            return eye_center, roll

        return None, None
