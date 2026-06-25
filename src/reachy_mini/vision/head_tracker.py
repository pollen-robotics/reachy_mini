"""Head tracker locating the nearest face with MediaPipe BlazeFace."""

from importlib.resources import files

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as e:
    raise ImportError(
        "mediapipe is not installed. Add the reachy_mini[vision] optional extra to your dependencies."
    ) from e

import numpy as np
from numpy.typing import NDArray

import reachy_mini

_MODEL_PATH = str(files(reachy_mini).joinpath("assets/blaze_face_short_range.tflite"))


class HeadTracker:
    """Locate the nearest face with MediaPipe BlazeFace to drive head pointing."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        """Initialize the head tracker."""
        self._detector = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
                min_detection_confidence=min_detection_confidence,
            )
        )

    def get_head_position(
        self, img: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float64], float] | tuple[None, None]:
        """Return eye-center (normalized to [-1, 1]) and roll (radians) of the nearest face in an RGB image."""
        detections = self._detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        ).detections
        if not detections:
            return None, None

        face = max(
            detections, key=lambda d: d.bounding_box.width * d.bounding_box.height
        )
        right_eye = np.array((face.keypoints[0].x, face.keypoints[0].y)) * 2 - 1
        left_eye = np.array((face.keypoints[1].x, face.keypoints[1].y)) * 2 - 1

        eye_center: NDArray[np.float64] = np.mean([right_eye, left_eye], axis=0)
        roll = float(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
        return eye_center, roll

    def close(self) -> None:
        """Release MediaPipe detector resources."""
        self._detector.close()
