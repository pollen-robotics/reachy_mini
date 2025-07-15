from reachy_mini import ReachyMini
from reachy_mini.placo_kinematics import PlacoKinematics
from reachy_mini.utils import create_head_pose


def offline_test_collision():
    head_kinematics = PlacoKinematics(
        urdf_path=ReachyMini.urdf_root_path
    )

    reachable_pose = create_head_pose()
    sol = head_kinematics.ik(reachable_pose, check_collision=True)
    assert sol is not None, "The reachable pose should not cause a collision."

    unreachable_pose = create_head_pose(x=20, y=20, mm=True)
    sol = head_kinematics.ik(unreachable_pose, check_collision=True)
    assert sol is None, "The unreachable pose should cause a collision."