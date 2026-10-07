"""Unit tests for the target fields of ``GET /state/full``.

The route reads the targets from the backend when asked with the
``with_target_*`` flags; ``FullState`` must carry them to the client
instead of silently dropping them.
"""

from __future__ import annotations

import numpy as np
import pytest

from reachy_mini.daemon.app.models import as_any_pose
from reachy_mini.daemon.app.routers import state
from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend


@pytest.fixture
def backend() -> MockupSimBackend:
    """Simulation backend: no audio, no hardware."""
    backend = MockupSimBackend(use_audio=False)
    # Seed the kinematics the way `run()` does on entry, so /state/full's
    # default pose fields resolve without spinning the control loop.
    backend.update_head_kinematics_model(
        backend._head_joint_positions,
        backend._antenna_joint_positions,
    )
    return backend


def test_full_state_returns_requested_targets(router_app, backend) -> None:
    """Each ``with_target_*`` flag brings its ``target_*`` field back.

    Targets are numpy arrays in the backend, as on the real robot, and
    use distinct non-zero values so a dropped or mixed-up field shows.
    """
    head_pose = np.eye(4)
    head_pose[:3, 3] = [0.01, -0.02, 0.03]
    backend.target_head_pose = head_pose
    backend.target_head_joint_positions = np.array(
        [0.7, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    backend.target_body_yaw = 0.3
    backend.target_antenna_joint_positions = np.array([0.1, -0.2])
    client = router_app(state.router, backend=backend)

    resp = client.get(
        "/state/full",
        params={
            "with_target_head_pose": True,
            "with_target_head_joints": True,
            "with_target_body_yaw": True,
            "with_target_antenna_positions": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["target_head_pose"] == as_any_pose(head_pose, False).model_dump()
    assert body["target_head_joints"] == [0.7, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert body["target_body_yaw"] == 0.3
    assert body["target_antennas_position"] == [0.1, -0.2]