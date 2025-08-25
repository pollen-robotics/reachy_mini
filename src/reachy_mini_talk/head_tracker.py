from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from insightface.app import FaceAnalysis

import logging

logger = logging.getLogger(__name__)


class HeadTracker:
    """
    Head Tracker using InsightFace to estimate head pose
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        providers: Optional[list[str]] = None,
    ) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(name=model_pack, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    # internals

    @staticmethod
    def _ensure_left_right(kps: np.ndarray) -> np.ndarray:
        """
        Ensure kps[0] is left eye and kps[1] is right eye by x-coordinate.
        """
        kps = np.asarray(kps, dtype=np.float32)
        if kps.shape[0] >= 2 and kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        return kps

    @staticmethod
    def _to_mp_coords(pt_px: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        Convert pixel coords -> normalized [0,1] -> MediaPipe-style [-1,1].
        """
        xy01 = np.array([pt_px[0] / float(w), pt_px[1] / float(h)], dtype=np.float32)
        return xy01 * 2.0 - 1.0

    @staticmethod
    def _pick_face(faces) -> Optional[object]:
        """
        Pick a single face similar to MediaPipe's max_num_faces=1:
        choose the highest detection score (ties broken by larger bbox area).
        """
        if not faces:
            return None
        best = None
        best_key = None
        for f in faces:
            score = float(getattr(f, "det_score", 0.0) or 0.0)
            bbox = getattr(f, "bbox", None)
            if bbox is not None and len(bbox) == 4:
                w = max(0.0, float(bbox[2] - bbox[0]))
                h = max(0.0, float(bbox[3] - bbox[1]))
                area = w * h
            else:
                area = 0.0
            key = (score, area)
            if best is None or key > best_key:
                best = f
                best_key = key
        return best

    # public API

    def get_eyes(
        self, img: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return (left_eye, right_eye) in MediaPipe-style [-1,1] coords.
        """
        faces = self.app.get(img)
        face = self._pick_face(faces)
        if face is None or getattr(face, "kps", None) is None:
            return None, None

        # attach image shape for downstream calls that rely on it
        h, w = img.shape[:2]
        setattr(face, "_img_shape", (h, w))

        kps = self._ensure_left_right(face.kps)

        left_eye = self._to_mp_coords(kps[0], w, h)
        right_eye = self._to_mp_coords(kps[1], w, h)
        return left_eye, right_eye

    def get_eyes_from_landmarks(self, face_landmarks) -> Tuple[np.ndarray, np.ndarray]:
        """
        return eyes in [-1,1], using a previously returned face object.
        """
        if getattr(face_landmarks, "kps", None) is None:
            raise ValueError("Face landmarks has no keypoints")

        img_shape = getattr(face_landmarks, "_img_shape", None)
        if img_shape is None:
            # Fallback to a safe default; but we strongly prefer having _img_shape set.
            h, w = 480, 640
        else:
            h, w = img_shape[:2]

        kps = self._ensure_left_right(face_landmarks.kps)
        left_eye = self._to_mp_coords(kps[0], w, h)
        right_eye = self._to_mp_coords(kps[1], w, h)
        return left_eye, right_eye

    def get_eye_center(self, face_landmarks) -> np.ndarray:
        """
        Average of left/right eye in [-1,1] space.
        """
        left_eye, right_eye = self.get_eyes_from_landmarks(face_landmarks)
        return np.mean([left_eye, right_eye], axis=0)

    def get_roll(self, face_landmarks) -> float:
        """
        Roll computed from the two eyes in [-1,1] space.
        """
        left_eye, right_eye = self.get_eyes_from_landmarks(face_landmarks)
        return float(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    def get_head_position(
        self, img: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Returns (eye_center [-1,1], roll).
        """
        faces = self.app.get(img)
        face = self._pick_face(faces)

        if face is None:
            logger.debug("No face detected.")
            return None, None

        score = float(getattr(face, "det_score", 1.0) or 1.0)
        if score < 0.3:  # permissive, like MediaPipe defaults you used
            logger.debug(f"Face detected but low score {score:.2f}.")
            return None, None

        if getattr(face, "kps", None) is None:
            logger.debug("Face detected but no keypoints found.")
            return None, None

        h, w = img.shape[:2]

        left_eye = face.kps[0]
        right_eye = face.kps[1]

        eye_center = np.mean([left_eye, right_eye], axis=0)

        eye_center_norm = np.array(
            [
                (eye_center[0] / w) * 2 - 1,  # x in [-1, 1]
                (eye_center[1] / h) * 2 - 1,  # y in [-1, 1]
            ]
        )

        # left_eye = self._to_mp_coords(kps[0], w, h)
        # right_eye = self._to_mp_coords(kps[1], w, h)

        # eye_center = np.mean([left_eye, right_eye], axis=0)
        roll = 0.0
        # roll = float(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        # return np.array([0.0, 0.0]), 0.0

        return eye_center_norm, roll

    def cleanup(self):
        if hasattr(self, "app"):
            del self.app
