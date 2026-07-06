"""Face detection with OpenCV's YuNet model."""

from dataclasses import dataclass

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray

_MODEL_REPO = "pollen-robotics/yunet-face-detection-2023mar"
_MODEL_FILE = "face_detection_yunet_2023mar.onnx"
_MODEL_REVISION = "664c75e50253bd3021bd976b497896149ef22c7b"


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
        model_path = hf_hub_download(_MODEL_REPO, _MODEL_FILE, revision=_MODEL_REVISION)
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold, nms_threshold
        )
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame_bgr: NDArray[np.uint8]) -> list[Face]:
        """Return every face detected in a BGR frame."""
        height, width = frame_bgr.shape[:2]
        # setInputSize regenerates the priors, so only call it when the size changes.
        if self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)
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
