"""Direct FFmpeg camera backend for local macOS applications."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from typing import Optional, cast

import cv2
import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_base import CameraBase
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
)


class FFmpegCamera(CameraBase):
    """Camera backend using an FFmpeg AVFoundation subprocess on macOS.

    FFmpeg can target AVFoundation devices by their exact display name, which
    avoids the unstable integer index mapping observed across different local
    capture stacks on macOS.
    """

    def __init__(
        self,
        device_name: str,
        log_level: str = "INFO",
        camera_specs: Optional[CameraSpecs] = None,
    ) -> None:
        super().__init__(log_level=log_level)
        self._device_name = device_name
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[npt.NDArray[np.uint8]] = None

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
        if self._process is not None:
            self.logger.info(
                "Restarting FFmpeg camera to apply resolution %s.", resolution.name
            )
            self.close()
            self.open()

    def _build_command(self) -> list[str]:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found in PATH")

        return [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-pixel_format",
            "bgr0",
            "-framerate",
            str(self.framerate),
            "-video_size",
            f"{self.resolution[0]}x{self.resolution[1]}",
            "-i",
            f"{self._device_name}:none",
            "-an",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "5",
            "pipe:1",
        ]

    def open(self) -> None:
        """Start the FFmpeg subprocess and warm it up."""
        if self._process is not None and self._process.poll() is None:
            return

        command = self._build_command()
        self.logger.info(
            "Opening direct FFmpeg camera fallback for device %r at %sx%s@%sfps.",
            self._device_name,
            self.resolution[0],
            self.resolution[1],
            self.framerate,
        )

        self._stop_event.clear()
        self._latest_frame = None
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = self._read_stderr()
                raise RuntimeError(
                    "FFmpeg camera process exited early while opening "
                    f"device {self._device_name!r}: {stderr}"
                )

            with self._frame_lock:
                if self._latest_frame is not None:
                    return
            time.sleep(0.05)

        raise RuntimeError(
            f"FFmpeg camera for device {self._device_name!r} did not deliver a frame."
        )

    def _reader_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        buffer = bytearray()
        while not self._stop_event.is_set():
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            buffer.extend(chunk)

            while True:
                soi = buffer.find(b"\xff\xd8")
                if soi == -1:
                    buffer.clear()
                    break

                eoi = buffer.find(b"\xff\xd9", soi + 2)
                if eoi == -1:
                    if soi > 0:
                        del buffer[:soi]
                    break

                jpeg = bytes(buffer[soi : eoi + 2])
                del buffer[: eoi + 2]

                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue

                with self._frame_lock:
                    self._latest_frame = cast(npt.NDArray[np.uint8], frame)

    def _read_stderr(self) -> str:
        process = self._process
        if process is None or process.stderr is None:
            return ""
        try:
            return process.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to read FFmpeg camera stderr."
            )
            return ""

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Return the latest decoded BGR frame."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def close(self) -> None:
        """Stop the FFmpeg subprocess and release resources."""
        self._stop_event.set()

        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3)
            self._process = None

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=3)
            self._reader_thread = None

        with self._frame_lock:
            self._latest_frame = None
