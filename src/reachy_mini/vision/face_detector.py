"""Face detection with OpenCV's YuNet model."""

import os
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from reachy_mini.utils.constants import ASSETS_ROOT_PATH

_MODEL_PATH = os.path.join(ASSETS_ROOT_PATH, "face_detection_yunet_2023mar.onnx")


@dataclass(frozen=True)
class Face:
    """A face detected in pixel coordinates: bounding box and eye centers."""

    bbox: tuple[float, float, float, float]
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]


class FaceDetector:
    """Detect faces in BGR frames with OpenCV's YuNet model."""

    def __init__(
        self, score_threshold: float = 0.6, nms_threshold: float = 0.3
    ) -> None:
        """Create the YuNet detector."""
        self._detector = cv2.FaceDetectorYN.create(
            _MODEL_PATH, "", (320, 320), score_threshold, nms_threshold
        )

    def detect(self, frame_bgr: NDArray[np.uint8]) -> list[Face]:
        """Return every face detected in a BGR frame."""
        height, width = frame_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []
        return [
            Face(
                bbox=(
                    float(faces[i, 0]),
                    float(faces[i, 1]),
                    float(faces[i, 2]),
                    float(faces[i, 3]),
                ),
                right_eye=(float(faces[i, 4]), float(faces[i, 5])),
                left_eye=(float(faces[i, 6]), float(faces[i, 7])),
            )
            for i in range(len(faces))
        ]
