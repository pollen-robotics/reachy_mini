import numpy as np
from numpy.testing import assert_almost_equal
from scipy.spatial.transform import Rotation as R

from reachy_mini.command import ReachyMiniCommand


def assert_command_equal(cmd1: ReachyMiniCommand, cmd2: ReachyMiniCommand):
    assert_almost_equal(cmd1.head_pose, cmd2.head_pose)
    assert_almost_equal(cmd1.antennas_orientation, cmd2.antennas_orientation)


def random_command() -> ReachyMiniCommand:
    """Generate a random command for testing."""
    roll, pitch, yaw = (
        2 * np.random.rand(3) - 1
    )  # Random values between -1 and 1 for roll, pitch, yaw
    head_rot = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()

    x, y, z = 0.1 * (
        np.random.rand(3) * 2 - 1
    )  # Random values between -0.1 and 0.1 for x, y, z

    head_pose = np.eye(4, dtype=np.float64)
    head_pose[:3, :3] = head_rot
    head_pose[:3, 3] = [x, y, z]

    # Antennas orientation: random values between -1 and 1
    antennas_orientation = 2 * np.random.rand(2) - 1

    return ReachyMiniCommand(
        head_pose=head_pose,
        antennas_orientation=antennas_orientation,
        offset_zero=True,
    )


def test_command_equality():
    """Test that two commands with the same values are equal."""
    cmd1 = random_command()
    cmd2 = cmd1.copy()
    assert_command_equal(cmd1, cmd2)
