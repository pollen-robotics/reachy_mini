"""Unit tests for daemon-side head-tracking backend plumbing."""

import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend
from reachy_mini.io.protocol import (
    GetTrackedFaceCmd,
    SetHeadTrackingCmd,
    command_adapter,
)


class DummyKinematics:
    """Kinematics spy that records the pose sent to IK."""

    def __init__(self) -> None:
        """Initialize spy state."""
        self.pose: np.ndarray | None = None

    def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
        """Record the IK input pose."""
        self.pose = pose
        return np.zeros(7, dtype=np.float64)


class DummyTracker:
    """Spy standing in for the face tracker."""

    def __init__(self) -> None:
        """Initialize spy state."""
        self.started = False
        self.stopped = False
        self.active_calls: list[bool] = []

    def start(self, camera_specs: object) -> None:
        """Record that the tracker was started."""
        self.started = True

    def set_active(self, active: bool) -> None:
        """Record pause/resume toggles."""
        self.active_calls.append(active)

    def latest(self) -> None:
        """Report no observation."""
        return None

    def stop(self) -> None:
        """Record that the tracker was stopped."""
        self.stopped = True


def _make_backend() -> MockupSimBackend:
    backend = MockupSimBackend(use_audio=False)
    backend.current_head_pose = np.eye(4, dtype=np.float64)
    return backend


def test_set_head_tracking_command_toggles_tracking() -> None:
    """The protocol command arms the gate and starts/stops the tracker."""
    backend = _make_backend()
    backend._media_server = SimpleNamespace(camera_specs=object())
    tracker = DummyTracker()
    backend._tracker = tracker

    responses: list[dict[str, object]] = []
    cmd = command_adapter.validate_python(
        {"type": "set_head_tracking", "enabled": True, "weight": 0.6}
    )
    backend.process_command(cmd, send_response=responses.append)

    assert isinstance(cmd, SetHeadTrackingCmd)
    assert backend._tracking_enabled is True
    assert backend._tracking_requested_weight == 0.6
    assert tracker.started is True
    assert tracker.active_calls[-1] is True
    assert responses[-1] == {
        "status": "ok",
        "command": "set_head_tracking",
        "enabled": True,
    }

    backend.process_command(
        SetHeadTrackingCmd(enabled=False), send_response=responses.append
    )

    assert backend._tracking_enabled is False
    assert tracker.stopped is True
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
        center=(0.25, -0.5),
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


def test_step_head_tracking_eases_toward_target() -> None:
    """The commanded aim glides toward the latched target rather than snapping."""
    backend = _make_backend()
    backend._tracking_enabled = True
    target = np.eye(4, dtype=np.float64)
    target[:3, :3] = R.from_euler("z", 0.5).as_matrix()
    backend._tracking_target_pose = target
    full = np.linalg.norm(R.from_matrix(target[:3, :3]).as_rotvec())

    backend.step_head_tracking()
    assert backend._tracking_aim is not None
    angle1 = np.linalg.norm(R.from_matrix(backend._tracking_aim[:3, :3]).as_rotvec())
    assert 0.0 < angle1 < full

    backend.step_head_tracking()
    angle2 = np.linalg.norm(R.from_matrix(backend._tracking_aim[:3, :3]).as_rotvec())
    assert angle1 < angle2 < full
    assert backend.ik_required is True


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


def test_tracking_face_loss_holds_last_target() -> None:
    """A transient face loss holds the target so the head doesn't lurch to neutral."""
    backend = _make_backend()
    backend._tracking_enabled = True
    target = np.eye(4, dtype=np.float64)
    backend._tracking_target_pose = target

    backend.set_tracking_face(
        center=None,
        roll=None,
        width=640,
        height=480,
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion=np.zeros(5, dtype=np.float64),
        timestamp=10.0,
    )

    assert backend.get_tracked_face().detected is False
    assert backend._tracking_target_pose is target


def test_tracking_sustained_loss_recenters_to_neutral() -> None:
    """After the lost timeout the aim target returns to the neutral head pose."""
    backend = _make_backend()
    backend._tracking_enabled = True
    target = np.eye(4, dtype=np.float64)
    target[:3, :3] = R.from_euler("z", 0.5).as_matrix()
    backend._tracking_target_pose = target
    backend._tracking_aim = target.copy()
    backend._last_face_seen = time.monotonic() - (backend._tracking_lost_timeout + 1.0)

    backend.step_head_tracking()

    np.testing.assert_allclose(backend._tracking_target_pose, np.eye(4), atol=1e-9)


def test_enable_head_tracking_weight_zero_pauses_without_stopping() -> None:
    """Weight 0 pauses the tracker and frees the head without stopping it."""
    backend = _make_backend()
    backend._media_server = SimpleNamespace(camera_specs=object())
    tracker = DummyTracker()
    backend._tracker = tracker
    backend._tracking_aim = np.eye(4, dtype=np.float64)
    backend._tracking_weight = 1.0

    backend.enable_head_tracking(weight=0.0)

    assert tracker.active_calls[-1] is False
    assert backend._tracking_weight == 0.0
    assert tracker.stopped is False
