import numpy as np
from placo_utils.tf import tf

import reachy_mini.analytic_kinematics as ak


def test_analytic_kinematics():
    ak_solver = ak.ReachyMiniAnalyticKinematics(
        urdf_path="src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
    )

    # initial joint angles
    joints = ak_solver.ik(ak_solver.T_world_head_home)
    assert np.allclose(
        np.array(list(joints.values())),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        atol=1e-2,
    ), "IK failed to return expected joint angles"


def test_with_a_pitch():
    ak_solver = ak.ReachyMiniAnalyticKinematics(
        urdf_path="src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
    )

    # Set a pitch angle
    pitch_angle = 0.2  # radians
    T_world_head = ak_solver.T_world_head_home.copy()

    T_world_head = tf.euler_matrix(0.0, pitch_angle, 0.0) @ T_world_head

    joints = ak_solver.ik(T_world_head)
    assert np.allclose(
        np.array(list(joints.values())),
        np.array([-0.43, -0.37, 0.26, -0.26, 0.37, 0.43]),
        atol=1e-2,
    ), "IK failed to return expected joint angles with pitch"


def test_with_a_yaw():
    ak_solver = ak.ReachyMiniAnalyticKinematics(
        urdf_path="src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
    )

    # Set a yaw angle
    yaw_angle = 0.2  # radians
    T_world_head = ak_solver.T_world_head_home.copy()

    T_world_head = tf.euler_matrix(0.0, 0.0, yaw_angle) @ T_world_head

    joints = ak_solver.ik(T_world_head)
    assert np.allclose(
        np.array(list(joints.values())),
        np.array([0.097, 0.079, 0.097, 0.079, 0.097, 0.079]),
        atol=1e-2,
    ), "IK failed to return expected joint angles with yaw"
