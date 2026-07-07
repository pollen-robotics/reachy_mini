"""Face detection with YuNet on ONNX Runtime (no OpenCV dependency)."""

import math
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray

_MODEL_REPO = "pollen-robotics/face_detection_yunet_2026may"
_MODEL_FILE = "face_detection_yunet_2026may.onnx"
_MODEL_REVISION = "2b8e922362946a0db67e861bae0f77826980effd"
_STRIDES = (8, 16, 32)
_MAX_STRIDE = 32


@dataclass(frozen=True)
class Face:
    """A face detected in pixel coordinates: bounding box and eye centers."""

    bbox: tuple[float, float, float, float]
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]


def _nms(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iou_threshold: float,
) -> list[int]:
    # Greedy IoU non-maximum suppression; returns kept indices, highest score first.
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    area = b[:, 2] * b[:, 3]
    order = np.array(scores, dtype=np.float32).argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        w = np.clip(
            np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0.0, None
        )
        h = np.clip(
            np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0.0, None
        )
        inter = w * h
        iou = inter / (area[i] + area[rest] - inter)
        order = rest[iou <= iou_threshold]
    return keep


class FaceDetector:
    """Detect faces in BGR frames with YuNet on ONNX Runtime."""

    def __init__(
        self, score_threshold: float = 0.6, nms_threshold: float = 0.3
    ) -> None:
        """Create the YuNet detector, confined to one CPU thread."""
        model_path = hf_hub_download(_MODEL_REPO, _MODEL_FILE, revision=_MODEL_REVISION)
        options = ort.SessionOptions()
        # Confine the detector to one thread so it can't spread across cores and starve the loop.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold

    def detect(self, frame_bgr: NDArray[np.uint8]) -> list[Face]:
        """Return every face detected in a BGR frame."""
        height, width = frame_bgr.shape[:2]
        # YuNet's feature-pyramid fusion needs stride-aligned dims; pad up (origin unchanged).
        padded_h = math.ceil(height / _MAX_STRIDE) * _MAX_STRIDE
        padded_w = math.ceil(width / _MAX_STRIDE) * _MAX_STRIDE
        if (padded_h, padded_w) != (height, width):
            padded = np.zeros((padded_h, padded_w, 3), dtype=frame_bgr.dtype)
            padded[:height, :width] = frame_bgr
            frame_bgr = padded
        blob = frame_bgr.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        outputs = dict(
            zip(self._output_names, self._session.run(None, {self._input_name: blob}))
        )
        return self._decode(outputs, padded_w)

    def _decode(
        self, outputs: dict[str, NDArray[np.float32]], width: int
    ) -> list[Face]:
        # Decode YuNet's per-stride heads into faces, then suppress overlaps.
        boxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []
        faces: list[Face] = []
        for stride in _STRIDES:
            cls = outputs[f"cls_{stride}"][0, :, 0]
            obj = outputs[f"obj_{stride}"][0, :, 0]
            # YuNet's confidence is the geometric mean of the classification and objectness heads.
            score = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
            idx = np.nonzero(score >= self._score_threshold)[0]
            if idx.size == 0:
                continue
            bbox = outputs[f"bbox_{stride}"][0][idx]
            kps = outputs[f"kps_{stride}"][0][idx]
            cols = width // stride
            col = (idx % cols).astype(np.float32)
            row = (idx // cols).astype(np.float32)
            cx = (col + bbox[:, 0]) * stride
            cy = (row + bbox[:, 1]) * stride
            w = np.exp(bbox[:, 2]) * stride
            h = np.exp(bbox[:, 3]) * stride
            for k in range(idx.size):
                box = (
                    float(cx[k] - w[k] / 2),
                    float(cy[k] - h[k] / 2),
                    float(w[k]),
                    float(h[k]),
                )
                boxes.append(box)
                scores.append(float(score[idx[k]]))
                faces.append(
                    Face(
                        bbox=box,
                        right_eye=(
                            float((col[k] + kps[k, 0]) * stride),
                            float((row[k] + kps[k, 1]) * stride),
                        ),
                        left_eye=(
                            float((col[k] + kps[k, 2]) * stride),
                            float((row[k] + kps[k, 3]) * stride),
                        ),
                    )
                )
        return [faces[i] for i in _nms(boxes, scores, self._nms_threshold)]
