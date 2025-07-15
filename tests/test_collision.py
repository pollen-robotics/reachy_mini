from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def test_collision():
    with ReachyMini() as mini:
        reachable_pose = create_head_pose()
        sol = mini.head_kinematics.ik(reachable_pose, check_collision=True)
        assert sol is not None, "The reachable pose should not cause a collision."

        unreachable_pose = create_head_pose(x=20, y=20, mm=True)
        sol = mini.head_kinematics.ik(unreachable_pose, check_collision=True)
        assert sol is None, "The unreachable pose should cause a collision."