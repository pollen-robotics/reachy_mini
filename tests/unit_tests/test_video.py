import time

import numpy as np
import numpy.typing as npt
import pytest

from reachy_mini.daemon.utils import is_local_camera_available
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
)
from reachy_mini.media.media_manager import MediaBackend, MediaManager
from reachy_mini.media.camera_base import CameraBase

SIGNALING_HOST = "reachy-mini.local"

# Video-capable backends to test
VIDEO_BACKENDS = [
    pytest.param(MediaBackend.LOCAL),
    pytest.param(MediaBackend.WEBRTC, marks=pytest.mark.wireless),
]


@pytest.mark.video
def test_is_local_camera_available(ipc_video_source: CameraSpecs) -> None:
    """Test that is_local_camera_available() detects the IPC endpoint.

    The ipc_video_source fixture creates a videotestsrc → IPC sink pipeline
    (unixfdsink on Linux/macOS, win32ipcvideosink on Windows).
    While it's running, ``is_local_camera_available()`` must return True.
    """
    assert is_local_camera_available(), (
        "is_local_camera_available() returned False while the IPC fixture is running"
    )


@pytest.mark.video
def test_get_frame_exists(ipc_video_source: CameraSpecs) -> None:
    """Test that a frame can be retrieved from the LOCAL IPC camera."""
    media = MediaManager(
        backend=MediaBackend.LOCAL,
        camera_specs=ipc_video_source,
    )
    time.sleep(1)
    frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert (
        frame.shape[0] == media.camera.resolution[1]
        and frame.shape[1] == media.camera.resolution[0]
    ), f"Frame has incorrect dimensions: {frame.shape} != {media.camera.resolution}"

    media.close()


@pytest.mark.video
@pytest.mark.wireless
def test_get_frame_exists_webrtc() -> None:
    """Test that a frame can be retrieved from the WebRTC camera."""
    media = MediaManager(
        backend=MediaBackend.WEBRTC,
        signalling_host=SIGNALING_HOST,
    )
    time.sleep(2)
    frame = media.get_frame()
    if frame is None:
        print("Waiting extra time for WebRTC backend to get the first frame...")
        time.sleep(4)
        frame = media.get_frame()
    assert frame is not None, "No frame was retrieved from the camera."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert (
        frame.shape[0] == media.camera.resolution[1]
        and frame.shape[1] == media.camera.resolution[0]
    ), f"Frame has incorrect dimensions: {frame.shape} != {media.camera.resolution}"

    media.close()


@pytest.mark.video
def test_get_frame_exists_all_resolutions(ipc_video_source: CameraSpecs) -> None:
    """Test that a frame can be retrieved at the default resolution.

    Note: changing resolution on the IPC reader does not change the
    daemon-side output resolution, so we only test that the default
    resolution works end-to-end.  Resolution validation logic is covered
    by ``test_change_resolution_errors``.
    """
    media = MediaManager(
        backend=MediaBackend.LOCAL,
        camera_specs=ipc_video_source,
    )
    time.sleep(1)
    frame = media.get_frame()
    resolution = ipc_video_source.default_resolution
    assert frame is not None, (
        f"No frame was retrieved from the camera at resolution {resolution}."
    )
    assert isinstance(frame, np.ndarray), (
        f"Frame is not a numpy array at resolution {resolution}."
    )
    assert frame.size > 0, f"Frame is empty at resolution {resolution}."
    assert (
        frame.shape[0] == resolution.value[1] and frame.shape[1] == resolution.value[0]
    ), f"Frame has incorrect dimensions at resolution {resolution}: {frame.shape}"

    media.close()


@pytest.mark.video
@pytest.mark.parametrize("backend", VIDEO_BACKENDS)
def test_change_resolution_errors(backend: MediaBackend) -> None:
    """Test that changing resolution raises errors for invalid specs/resolutions."""
    media = MediaManager(
        backend=backend,
        signalling_host=SIGNALING_HOST
        if backend == MediaBackend.WEBRTC
        else "localhost",
    )
    media.camera.camera_specs = None
    with pytest.raises(RuntimeError):
        media.camera.set_resolution(CameraResolution.R1280x720at30fps)

    # TODO: uncomment this when we actually support resolution change for Mujoco cameras. 
    # We currently only log a warning to avoid raising an error when the camera is initialized in `init_camera()`
    # media.camera.camera_specs = MujocoCameraSpecs()
    # with pytest.raises(RuntimeError):
    #     media.camera.set_resolution(CameraResolution.R1280x720at30fps)

    media.camera.camera_specs = ReachyMiniLiteCamSpecs()
    with pytest.raises(ValueError):
        media.camera.set_resolution(CameraResolution.R1280x720at30fps)

    media.close()


@pytest.mark.video
def test_read_jpeg_encodes_bgr_frame() -> None:
    """CameraBase.read_jpeg encodes a BGR frame into a JPEG that decodes back to it."""
    cv2 = pytest.importorskip("cv2")

    class _StubCamera(CameraBase):
        def open(self) -> None: ...

        def read(self) -> npt.NDArray[np.uint8]:
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[:, :, 2] = 255  # BGR red
            return frame

        def close(self) -> None:
            self._release_jpeg_encoder()

        def _apply_resolution(self, resolution: CameraResolution) -> None: ...

    camera = _StubCamera()
    try:
        jpeg = camera.read_jpeg()
    finally:
        camera.close()

    assert jpeg is not None
    assert jpeg[:2] == b"\xff\xd8"

    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    blue, green, red = (int(c) for c in decoded[16, 16])
    assert red > 200 and green < 40 and blue < 40
