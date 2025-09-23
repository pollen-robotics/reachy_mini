from reachy_mini.utils.constants import URDF_ROOT_PATH
from reachy_mini.kinematics import PlacoKinematics
from reachy_mini.utils import create_head_pose


def offline_test_collision():
    head_kinematics = PlacoKinematics(
        urdf_path=URDF_ROOT_PATH, check_collision=True
    )

    reachable_pose = create_head_pose()
    sol = head_kinematics.ik(reachable_pose)
    assert sol is not None, "The reachable pose should not cause a collision."

    unreachable_pose = create_head_pose(x=20, y=20, mm=True)
    sol = head_kinematics.ik(unreachable_pose)
    assert sol is None, "The unreachable pose should cause a collision."
