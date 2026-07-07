"""Tests for the YuNet face detector (ONNX Runtime)."""

import math

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from reachy_mini.vision import face_detector


def _detector(score_threshold: float = 0.6) -> "face_detector.FaceDetector":
    try:
        return face_detector.FaceDetector(score_threshold=score_threshold)
    except Exception as exc:  # model repo unreachable / offline
        pytest.skip(f"YuNet model unavailable: {exc}")


def test_detect_returns_empty_on_blank_frame() -> None:
    """A frame with no faces yields no detections."""
    blank = np.zeros((180, 320, 3), dtype=np.uint8)
    assert _detector().detect(blank) == []


def test_matches_opencv_reference() -> None:
    """The ONNX Runtime decode reproduces cv2.FaceDetectorYN on the same model and frame."""
    cv2 = pytest.importorskip("cv2")
    from huggingface_hub import hf_hub_download

    score = 0.015  # low, to force detections on a synthetic (faceless) frame
    detector = _detector(score)

    width, height = 320, 180
    frame = cv2.blur(
        np.random.default_rng(5).integers(0, 255, (height, width, 3), dtype=np.uint8),
        (15, 15),
    )
    ours = detector.detect(frame)

    # cv2 running the same model on the same /32-padded frame is the reference decode.
    padded_w = math.ceil(width / 32) * 32
    padded_h = math.ceil(height / 32) * 32
    padded = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
    padded[:height, :width] = frame
    model = hf_hub_download(
        face_detector._MODEL_REPO,
        face_detector._MODEL_FILE,
        revision=face_detector._MODEL_REVISION,
    )
    reference = cv2.FaceDetectorYN.create(model, "", (padded_w, padded_h), score, 0.3)
    reference.setInputSize((padded_w, padded_h))
    _, cv_faces = reference.detect(padded)
    cv_faces = np.empty((0, 15), np.float32) if cv_faces is None else cv_faces

    assert len(ours) == len(cv_faces)
    for face, ref in zip(ours, cv_faces):
        assert np.allclose(face.bbox, ref[:4], atol=1e-2)
        assert np.allclose(face.right_eye, ref[4:6], atol=1e-2)
        assert np.allclose(face.left_eye, ref[6:8], atol=1e-2)
