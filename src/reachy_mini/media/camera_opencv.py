"""Direct OpenCV camera backend for local macOS applications."""

from __future__ import annotations

import time
from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_base import CameraBase
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
)

cv2: Any | None
try:
    import cv2 as _cv2
except ImportError:
    cv2 = None
else:
    cv2 = _cv2


class OpenCVCamera(CameraBase):
    """Camera backend using OpenCV + AVFoundation directly."""

    def __init__(
        self,
        device_index: int,
        log_level: str = "INFO",
        camera_specs: Optional[CameraSpecs] = None,
    ) -> None:
        """Initialize the direct OpenCV camera backend."""
        super().__init__(log_level=log_level)
        if cv2 is None:
            raise ImportError(
                "The 'cv2' module is required for OpenCVCamera but could not be imported."
            )

        self._device_index = device_index
        self._capture: Any | None = None

        if camera_specs is not None:
            self.camera_specs = camera_specs
        else:
            self.logger.warning(
                "No camera_specs provided — defaulting to ReachyMiniLiteCamSpecs."
            )
            self.camera_specs = ReachyMiniLiteCamSpecs()
        self._resolution = self.camera_specs.default_resolution
        self.resized_K = self.camera_specs.K

    def _apply_resolution(self, resolution: CameraResolution) -> None:
        self._resolution = resolution
        if self._capture is None:
            return

        cv2_module = self._get_cv2()
        self._capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._capture.set(cv2_module.CAP_PROP_FPS, self.framerate)

    @staticmethod
    def _get_cv2() -> Any:
        if cv2 is None:
            raise RuntimeError("OpenCV is not available.")
        return cv2

    def open(self) -> None:
        """Open the AVFoundation camera and warm it up."""
        if self._capture is not None and self._capture.isOpened():
            return

        cv2_module = self._get_cv2()
        capture = cv2_module.VideoCapture(
            self._device_index, cv2_module.CAP_AVFOUNDATION
        )
        if not capture.isOpened():
            raise RuntimeError(
                f"Failed to open AVFoundation camera at index {self._device_index}"
            )

        self._capture = capture
        resolution = self._resolution
        if resolution is None:
            raise RuntimeError("Camera resolution is not set.")
        self._apply_resolution(resolution)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None:
                return
            time.sleep(0.05)

        self.logger.warning(
            "OpenCV camera opened but did not deliver a frame within the warm-up window."
        )

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame directly from the camera."""
        capture = self._capture
        if capture is None or not capture.isOpened():
            return None

        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        return cast(npt.NDArray[np.uint8], frame)

    def close(self) -> None:
        """Release the OpenCV capture handle."""
        if self._capture is None:
            return
        self._capture.release()
        self._capture = None
