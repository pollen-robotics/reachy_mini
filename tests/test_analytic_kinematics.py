import numpy as np
from placo_utils.tf import tf

import reachy_mini.analytic_kinematics as ak

ak_solver = ak.ReachyMiniAnalyticKinematics(
    urdf_path="src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
)

def test_analytic_kinematics():

    # initial joint angles
    joints = ak_solver.ik(np.eye(4), body_yaw=0.0)
    assert np.allclose(
        np.array(joints),
        np.array([ 0. ,  0.54690032, -0.69118484,  0.62905114, -0.62905434,  0.69118874, -0.54690758]),
        atol=1e-2,
    ), "IK failed to return expected joint angles"


def test_with_a_pitch():

    # Set a pitch angle
    pitch_angle = 0.2  # radians
    pose = tf.euler_matrix(0.0, pitch_angle, 0.0, axes="sxyz")

    joints = ak_solver.ik(pose, body_yaw=0.0)
    assert np.allclose(
        np.array(joints),
        np.array([ 0. ,  0.40565672, -0.59685794,  0.75626897, -0.75627194, 0.59686182, -0.40566356]),
        atol=1e-2,
    ), "IK failed to return expected joint angles with pitch"


def test_with_a_yaw():

    # Set a yaw angle
    yaw_angle = 0.2  # radians
    pose = tf.euler_matrix(0.0, 0.0, yaw_angle, axes="sxyz")

    joints = ak_solver.ik(pose, body_yaw=0.0)
    assert np.allclose(
        np.array(joints),
        np.array([0. ,  0.62135209, -0.60954831,  0.68185994, -0.58819444, 0.78379322, -0.49168039]),
        atol=1e-2,
    ), "IK failed to return expected joint angles with yaw"


def test_z_axis():
    
    # Set a target position along the Z-axis
    target_position = np.array([-0.008, 0.0, -0.025])
    pose = tf.translation_matrix(target_position)

    joints = ak_solver.ik(pose, body_yaw=0.0)
    assert np.allclose(
        np.array(joints),
        np.array([0.0]*7),
        atol=1e-2,
    ), "IK failed to return expected joint angles for Z-axis target"
    
def test_body_yaw():

    # Set a body yaw angle
    body_yaw = 0.1  # radians
    target_position = np.array([-0.008, 0.0, -0.025])
    pose = tf.translation_matrix(target_position) @ tf.euler_matrix(0.0, 0.0, body_yaw, axes="sxyz")

    joints = ak_solver.ik(pose, body_yaw=-body_yaw)
    assert np.allclose(
        np.array(joints),
        np.array([-body_yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        atol=1e-2,
    ), "IK failed to return expected joint angles with body yaw"
