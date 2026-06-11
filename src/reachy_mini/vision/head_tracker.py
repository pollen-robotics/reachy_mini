"""Head Tracker using MediaPipe Face Mesh to estimate head pose."""

from collections.abc import Sequence
from typing import Protocol

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("mediapipe is not installed. Add the reachy_mini[vision] optional extra to your dependencies.") from e

import numpy as np
from numpy.typing import NDArray


class _Landmark(Protocol):
    x: float
    y: float


class HeadTracker:
    """Head Tracker using MediaPipe Face Mesh to estimate head pose."""

    def __init__(self) -> None:
        """Initialize the Head Tracker."""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.05,
            min_tracking_confidence=0.5,
            max_num_faces=1,
        )

    def get_eyes(
        self, img: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[None, None]:
        """Get the coordinates of the eyes from the image."""
        results = self.face_mesh.process(img)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_eye = np.array((landmarks[33].x, landmarks[33].y))
            left_eye = left_eye * 2 - 1

            right_eye = np.array((landmarks[263].x, landmarks[263].y))
            right_eye = right_eye * 2 - 1
            return left_eye, right_eye

        return None, None

    def _get_eyes_from_landmarks(
        self, landmarks: Sequence[_Landmark]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the coordinates of the eyes from face landmarks."""
        left_eye = np.array((landmarks[33].x, landmarks[33].y))
        left_eye = left_eye * 2 - 1

        right_eye = np.array((landmarks[263].x, landmarks[263].y))
        right_eye = right_eye * 2 - 1
        return left_eye, right_eye

    def _get_eye_center(self, landmarks: Sequence[_Landmark]) -> NDArray[np.float64]:
        """Get the center of the eyes from face landmarks."""
        left_eye, right_eye = self._get_eyes_from_landmarks(landmarks)
        eye_center: NDArray[np.float64] = np.mean([left_eye, right_eye], axis=0)
        return eye_center

    def _get_roll(self, landmarks: Sequence[_Landmark]) -> np.float64:
        """Calculate the roll of the head based on eye positions."""
        left_eye = np.array((landmarks[33].x, landmarks[33].y))
        left_eye = left_eye * 2 - 1

        right_eye = np.array((landmarks[263].x, landmarks[263].y))
        right_eye = right_eye * 2 - 1
        roll: np.float64 = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

        return roll

    def get_head_position(
        self, img: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float64], np.float64] | tuple[None, None]:
        """Get the head position and roll from the image."""
        results = self.face_mesh.process(img)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            eye_center = self._get_eye_center(landmarks)  # [-1, 1] [-1, 1]
            roll = self._get_roll(landmarks)

            return eye_center, roll

        return None, None
