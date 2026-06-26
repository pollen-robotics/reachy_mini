"""Unit tests for daemon-side head-tracking backend plumbing."""

from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend
from reachy_mini.io.protocol import (
    GetTrackedFaceCmd,
    SetHeadTrackingCmd,
    command_adapter,
)
from reachy_mini.vision.look_at import look_at_image_pose


class DummyKinematics:
    """Kinematics spy that records the pose sent to IK."""

    def __init__(self) -> None:
        """Initialize spy state."""
        self.pose: np.ndarray | None = None

    def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
        """Record the IK input pose."""
        self.pose = pose
        return np.zeros(7, dtype=np.float64)


class DummyMediaServer:
    """Small media-server spy for head-tracking command tests."""

    def __init__(self) -> None:
        """Initialize spy state."""
        self.enabled = False
        self.disabled = False
        self.callback: Callable[..., None] | None = None

    def enable_tracking(self, callback: Callable[..., None]) -> bool:
        """Record that tracking was enabled."""
        self.enabled = True
        self.callback = callback
        return True

    def disable_tracking(self) -> None:
        """Record that tracking was disabled."""
        self.disabled = True


def _make_backend() -> MockupSimBackend:
    backend = MockupSimBackend(use_audio=False)
    backend.current_head_pose = np.eye(4, dtype=np.float64)
    return backend


def test_set_head_tracking_command_toggles_media_server() -> None:
    """The protocol command should mirror wobbling's media-server hook pattern."""
    backend = _make_backend()
    media = DummyMediaServer()
    backend._media_server = media

    responses: list[dict[str, object]] = []
    cmd = command_adapter.validate_python(
        {"type": "set_head_tracking", "enabled": True, "weight": 0.6}
    )
    backend.process_command(cmd, send_response=responses.append)

    assert isinstance(cmd, SetHeadTrackingCmd)
    assert media.enabled is True
    assert media.callback == backend.set_tracking_face
    assert responses[-1] == {
        "status": "ok",
        "command": "set_head_tracking",
        "enabled": True,
    }

    backend.process_command(
        SetHeadTrackingCmd(enabled=False), send_response=responses.append
    )

    assert media.disabled is True
    assert responses[-1] == {
        "status": "ok",
        "command": "set_head_tracking",
        "enabled": False,
    }


def test_get_tracked_face_command_returns_latest_face() -> None:
    """The query command returns the latest normalized face target."""
    backend = _make_backend()
    backend._tracking_enabled = True
    backend.set_tracking_face(
        eye_center=np.array([0.25, -0.5], dtype=np.float64),
        roll=0.1,
        width=640,
        height=480,
        camera_matrix=np.array(
            [[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion=np.zeros(5, dtype=np.float64),
        timestamp=123.0,
    )

    responses: list[dict[str, object]] = []
    backend.process_command(GetTrackedFaceCmd(), send_response=responses.append)

    assert responses == [
        {
            "command": "get_tracked_face",
            "face_target": {
                "detected": True,
                "x": 0.25,
                "y": -0.5,
                "roll": 0.1,
                "ts": 123.0,
            },
        }
    ]


def test_tracking_blend_respects_zero_and_full_weight() -> None:
    """Tracking blend should be identity at 0 and full aim at 1."""
    backend = _make_backend()
    kinematics = DummyKinematics()
    backend.head_kinematics = kinematics
    app_pose = np.eye(4, dtype=np.float64)
    aim = np.eye(4, dtype=np.float64)
    aim[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    backend._tracking_aim = aim

    backend._tracking_weight = 0.0
    backend.update_target_head_joints_from_ik(app_pose)
    assert kinematics.pose is not None
    np.testing.assert_allclose(kinematics.pose, app_pose)

    backend._tracking_weight = 1.0
    backend.update_target_head_joints_from_ik(app_pose)
    assert kinematics.pose is not None
    np.testing.assert_allclose(kinematics.pose, aim, atol=1e-12)


def test_full_weight_tracking_ignores_unrelated_app_head_updates() -> None:
    """App head targets should not dirty IK while full tracking owns the head."""
    backend = _make_backend()
    app_pose = np.eye(4, dtype=np.float64)
    app_pose[0, 3] = 0.05
    backend._tracking_aim = np.eye(4, dtype=np.float64)
    backend._tracking_weight = 1.0

    backend.ik_required = False
    backend.set_target_head_pose(app_pose)
    assert backend.target_head_pose is app_pose
    assert backend.ik_required is False

    backend.target_body_yaw = 0.0
    backend.set_target_body_yaw(0.0)
    assert backend.ik_required is False


def test_tracking_face_loss_holds_last_aim() -> None:
    """A transient face loss holds the aim so the head doesn't lurch to neutral."""
    backend = _make_backend()
    backend._tracking_enabled = True
    aim = np.eye(4, dtype=np.float64)
    backend._tracking_aim = aim
    backend._tracking_weight = 1.0

    backend.set_tracking_face(
        eye_center=None,
        roll=None,
        width=640,
        height=480,
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion=np.zeros(5, dtype=np.float64),
        timestamp=10.0,
    )

    assert backend.get_tracked_face().detected is False
    assert backend._tracking_weight == 1.0
    assert backend._tracking_aim is aim


def test_tracking_face_updates_are_smoothed() -> None:
    """Tracking should smooth detector jumps before updating the aim."""
    backend = _make_backend()
    backend._tracking_enabled = True
    camera_matrix = np.array(
        [[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)

    backend.set_tracking_face(
        eye_center=np.array([0.0, 0.0], dtype=np.float64),
        roll=0.0,
        width=640,
        height=480,
        camera_matrix=camera_matrix,
        distortion=distortion,
        timestamp=1.0,
    )
    backend.set_tracking_face(
        eye_center=np.array([1.0, 0.0], dtype=np.float64),
        roll=0.0,
        width=640,
        height=480,
        camera_matrix=camera_matrix,
        distortion=distortion,
        timestamp=1.1,
    )

    assert backend.get_tracked_face().x == 0.25


def test_tracking_aim_eases_toward_face_instead_of_snapping() -> None:
    """The stored aim turns toward the face but by less than the full look-at.

    Snapping to the absolute aim (gain 1.0) overshoots under capture-to-actuation
    latency and oscillates; easing keeps the head stable.
    """
    backend = _make_backend()
    backend._tracking_enabled = True
    camera_matrix = np.array(
        [[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)
    backend.set_tracking_face(
        eye_center=np.array([0.6, 0.0], dtype=np.float64),
        roll=0.0,
        width=640,
        height=480,
        camera_matrix=camera_matrix,
        distortion=distortion,
        timestamp=1.0,
    )

    full_aim = look_at_image_pose(
        u=(0.6 + 1.0) * 0.5 * 639,
        v=(0.0 + 1.0) * 0.5 * 479,
        K=camera_matrix,
        D=distortion,
        T_world_head=np.eye(4, dtype=np.float64),
        T_head_cam=backend.T_head_cam,
    )
    assert backend._tracking_aim is not None
    eased = np.linalg.norm(R.from_matrix(backend._tracking_aim[:3, :3]).as_rotvec())
    full = np.linalg.norm(R.from_matrix(full_aim[:3, :3]).as_rotvec())
    assert 0.0 < eased < full
