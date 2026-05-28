"""Standalone placo/pinocchio test — no reachy_mini imports.

This isolates whether placo works on its own in the pytest environment,
without any reachy_mini import chain (media_manager, GStreamer, etc.).
"""

import os
import platform

import pytest

placo = pytest.importorskip("placo", reason="Placo is not available on this platform")


def test_placo_load_urdf():
    """Load the URDF with placo directly, no reachy_mini imports."""
    urdf_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "reachy_mini",
        "descriptions",
        "reachy_mini",
        "urdf",
    )
    urdf_path = os.path.join(urdf_dir, "robot_no_collision.urdf")
    assert os.path.exists(urdf_path), f"URDF not found: {urdf_path}"

    robot = placo.RobotWrapper(
        urdf_path, placo.Flags.collision_as_visual + placo.Flags.ignore_collisions
    )
    assert robot is not None
    assert len(robot.joint_names()) > 0


def test_placo_fk_solver():
    """Create a placo FK solver and run one solve step."""
    import numpy as np

    urdf_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "reachy_mini",
        "descriptions",
        "reachy_mini",
        "urdf",
    )
    urdf_path = os.path.join(urdf_dir, "robot_no_collision.urdf")

    robot = placo.RobotWrapper(
        urdf_path, placo.Flags.collision_as_visual + placo.Flags.ignore_collisions
    )
    solver = placo.KinematicsSolver(robot)
    solver.mask_fbase(True)
    solver.enable_velocity_limits(True)
    solver.enable_joint_limits(True)
    solver.dt = 0.02

    # Add a joints task with zero targets
    joints_task = solver.add_joints_task()
    joints_task.set_joints(
        {
            "yaw_body": 0.0,
            "stewart_1": 0.0,
            "stewart_2": 0.0,
            "stewart_3": 0.0,
            "stewart_4": 0.0,
            "stewart_5": 0.0,
            "stewart_6": 0.0,
        }
    )
    joints_task.configure("joints", "soft", 5.0)

    # Add closing loop constraints
    for i in range(1, 6):
        task = solver.add_relative_position_task(
            f"closing_{i}_1", f"closing_{i}_2", np.zeros(3)
        )
        task.configure(f"closing_{i}", "hard", 1.0)

    # Solve
    solver.solve(True)
    robot.update_kinematics()

    # Should be able to read the head frame
    T = robot.get_T_world_frame("head")
    assert T.shape == (4, 4)
