from reachy_mini.kinematics import AnalyticalKinematics
import numpy as np

def test_analytical_kinematics():
    ak = AnalyticalKinematics()
    pose = np.eye(4)
    sol = ak.ik(pose)
    assert sol is not None, "IK solution should be found"
    fk_pose = ak.fk(sol, no_iterations=10)
    assert np.allclose(fk_pose, pose, atol=1e-2), "FK should match the original pose"
    
def test_analytical_kinematics_with_yaw():
    ak = AnalyticalKinematics()
    pose = np.eye(4)
    body_yaw = np.pi / 4  # 45 degrees
    sol = ak.ik(pose, body_yaw=body_yaw)
    assert sol is not None, "IK solution should be found with body yaw"
    fk_pose = ak.fk(sol, no_iterations=10)
    assert np.allclose(fk_pose, pose, atol=1e-2), "FK should match the original pose with body yaw"