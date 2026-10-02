"""Pin reachy_mini.utils.rotation.Rotation to scipy's behaviour.

The SDK dropped scipy from its import path because
`from scipy.spatial.transform import Rotation` costs ~1.7 s of cold import on
the robot's CM4. scipy stays installed purely as the oracle here: every
operation the SDK uses is fuzzed against it, so an upstream behaviour change
or a mistake in our implementation surfaces in CI rather than on a robot.

Tolerances are absolute. 1e-12 is already several orders of magnitude below
anything the hardware can express (the head resolves far worse than a
microradian), so it leaves plenty of headroom while still catching a genuinely
wrong formula.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

from reachy_mini.utils.rotation import Rotation

TOL = 1e-12
N = 500

# Head pitch/roll are clamped to +/-40 degrees, so gimbal lock (pitch = +/-90)
# is unreachable. The fuzz still covers all of SO(3) to keep the maths honest
# for any future call site.
REALISTIC_RPY_DEG = (40.0, 40.0, 180.0)


@pytest.fixture
def rng():
    """Deterministic generator, so a failure is always reproducible."""
    return np.random.default_rng(20260811)


def _random_matrices(rng, count):
    return ScipyRotation.random(
        count, random_state=int(rng.integers(0, 2**31))
    ).as_matrix()


def test_from_euler_xyz_radians(rng):
    """Extrinsic xyz from radians, the convention the whole SDK assumes."""
    for angles in rng.uniform(-np.pi, np.pi, size=(N, 3)):
        assert np.allclose(
            ScipyRotation.from_euler("xyz", angles).as_matrix(),
            Rotation.from_euler("xyz", angles).as_matrix(),
            atol=TOL,
        )


def test_from_euler_xyz_degrees(rng):
    """The degrees=True path, used for the 20 degree tilt of the rest pose."""
    for angles in rng.uniform(-180, 180, size=(N, 3)):
        assert np.allclose(
            ScipyRotation.from_euler("xyz", angles, degrees=True).as_matrix(),
            Rotation.from_euler("xyz", angles, degrees=True).as_matrix(),
            atol=TOL,
        )


def test_from_euler_single_axis_scalar(rng):
    """interpolation.py factors yaw out with a scalar single-axis sequence."""
    for angle in rng.uniform(-np.pi, np.pi, size=N):
        for axis in "xyz":
            assert np.allclose(
                ScipyRotation.from_euler(axis, angle).as_matrix(),
                Rotation.from_euler(axis, angle).as_matrix(),
                atol=TOL,
            )


def test_as_euler_xyz(rng):
    """Matrix to extrinsic xyz, including the branch cuts."""
    for matrix in _random_matrices(rng, N):
        assert np.allclose(
            ScipyRotation.from_matrix(matrix).as_euler("xyz"),
            Rotation.from_matrix(matrix).as_euler("xyz"),
            atol=TOL,
        )


def test_as_euler_over_the_reachable_workspace(rng):
    """The angles the robot can actually reach get the same treatment."""
    limits = np.array(REALISTIC_RPY_DEG)
    for angles in rng.uniform(-limits, limits, size=(N, 3)):
        matrix = ScipyRotation.from_euler("xyz", angles, degrees=True).as_matrix()
        assert np.allclose(
            ScipyRotation.from_matrix(matrix).as_euler("xyz"),
            Rotation.from_matrix(matrix).as_euler("xyz"),
            atol=TOL,
        )


def test_as_rotvec(rng):
    """Matrix to rotation vector, the SLERP ingredient."""
    for matrix in _random_matrices(rng, N):
        assert np.allclose(
            ScipyRotation.from_matrix(matrix).as_rotvec(),
            Rotation.from_matrix(matrix).as_rotvec(),
            atol=TOL,
        )


def test_from_rotvec_including_degenerate_magnitudes(rng):
    """Rodrigues at the awkward magnitudes: ~0, ~pi and exactly pi."""
    for axis in rng.normal(size=(N, 3)):
        axis = axis / np.linalg.norm(axis)
        for theta in (rng.uniform(0, np.pi), 0.0, 1e-12, 1e-9, np.pi - 1e-9, np.pi):
            rotvec = axis * theta
            assert np.allclose(
                ScipyRotation.from_rotvec(rotvec).as_matrix(),
                Rotation.from_rotvec(rotvec).as_matrix(),
                atol=TOL,
            ), f"axis={axis} theta={theta}"


def test_composition_and_inverse(rng):
    """`*` composes in scipy's order and `.inv()` undoes it."""
    for a, b in zip(_random_matrices(rng, N), _random_matrices(rng, N)):
        sa, sb = ScipyRotation.from_matrix(a), ScipyRotation.from_matrix(b)
        ma, mb = Rotation.from_matrix(a), Rotation.from_matrix(b)
        assert np.allclose((sa * sb).as_matrix(), (ma * mb).as_matrix(), atol=TOL)
        assert np.allclose(sa.inv().as_matrix(), ma.inv().as_matrix(), atol=TOL)
        assert np.allclose(
            (sa.inv() * sb).as_rotvec(), (ma.inv() * mb).as_rotvec(), atol=TOL
        )


def test_shapes_and_types_match_scipy(rng):
    """A (1,3,3) for a (3,3) would slip through allclose but break callers."""
    matrix = _random_matrices(rng, 1)[0]
    pairs = [
        (
            ScipyRotation.from_matrix(matrix).as_matrix(),
            Rotation.from_matrix(matrix).as_matrix(),
        ),
        (
            ScipyRotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix(),
            Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix(),
        ),
        (
            ScipyRotation.from_euler("z", 0.4).as_matrix(),
            Rotation.from_euler("z", 0.4).as_matrix(),
        ),
        (
            ScipyRotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix(),
            Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix(),
        ),
        (
            ScipyRotation.from_matrix(matrix).as_euler("xyz"),
            Rotation.from_matrix(matrix).as_euler("xyz"),
        ),
        (
            ScipyRotation.from_matrix(matrix).as_rotvec(),
            Rotation.from_matrix(matrix).as_rotvec(),
        ),
    ]
    for expected, actual in pairs:
        assert np.shape(expected) == np.shape(actual)
        assert np.asarray(expected).dtype == np.asarray(actual).dtype

    # models.py unpacks as_euler into three scalars, so element type matters.
    assert isinstance(Rotation.from_matrix(matrix).as_euler("xyz")[2], np.float64)


def test_linear_pose_interpolation_matches_a_scipy_reference(rng):
    """End to end on the real consumer, both branches, over the whole path.

    This is the function that actually runs in the goto and recorded-move
    loops, so it gets compared against a scipy-backed reimplementation of
    itself rather than only unit-tested piecewise.
    """
    from reachy_mini.utils.interpolation import linear_pose_interpolation

    def scipy_reference(start, target, t, yaw_as_scalar):
        rot_start = ScipyRotation.from_matrix(start[:3, :3])
        rot_end = ScipyRotation.from_matrix(target[:3, :3])
        if yaw_as_scalar:
            yaw_start = rot_start.as_euler("xyz")[2]
            yaw_end = rot_end.as_euler("xyz")[2]
            yaw_interp = yaw_start + (yaw_end - yaw_start) * t
            res_start = ScipyRotation.from_euler("z", -yaw_start) * rot_start
            res_end = ScipyRotation.from_euler("z", -yaw_end) * rot_end
            rotvec_rel = (res_start.inv() * res_end).as_rotvec()
            res_interp = res_start * ScipyRotation.from_rotvec(rotvec_rel * t)
            rot_interp = (
                ScipyRotation.from_euler("z", yaw_interp) * res_interp
            ).as_matrix()
        else:
            rotvec_rel = (rot_start.inv() * rot_end).as_rotvec()
            rot_interp = (
                rot_start * ScipyRotation.from_rotvec(rotvec_rel * t)
            ).as_matrix()
        out = np.eye(4)
        out[:3, :3] = rot_interp
        out[:3, 3] = start[:3, 3] + (target[:3, 3] - start[:3, 3]) * t
        return out

    for a, b in zip(_random_matrices(rng, 60), _random_matrices(rng, 60)):
        start, target = np.eye(4), np.eye(4)
        start[:3, :3], target[:3, :3] = a, b
        start[:3, 3] = rng.uniform(-0.05, 0.05, 3)
        target[:3, 3] = rng.uniform(-0.05, 0.05, 3)
        for t in (0.0, 0.17, 0.5, 0.83, 1.0):
            for yaw_as_scalar in (False, True):
                assert np.allclose(
                    scipy_reference(start, target, t, yaw_as_scalar),
                    linear_pose_interpolation(start, target, t, yaw_as_scalar),
                    atol=1e-10,
                )


def test_from_matrix_orthonormalises_like_scipy(rng):
    """Slightly drifted matrices must land on the same nearest rotation."""
    for matrix in _random_matrices(rng, N):
        drifted = matrix + rng.normal(scale=1e-6, size=(3, 3))
        assert np.allclose(
            ScipyRotation.from_matrix(drifted).as_matrix(),
            Rotation.from_matrix(drifted).as_matrix(),
            atol=1e-9,
        )


def test_as_euler_at_exact_gimbal_lock(rng):
    """Pitch = +/-90 deg, the one region random fuzzing cannot land on.

    Random rotations never fall within the lock threshold, so this branch
    needs its own cases: scipy pins the third angle to 0 there and we must
    pick the exact same representative.
    """
    import warnings

    for sign in (1.0, -1.0):
        for alpha, gamma in rng.uniform(-np.pi, np.pi, size=(50, 2)):
            matrix = ScipyRotation.from_euler(
                "xyz", [alpha, sign * np.pi / 2, gamma]
            ).as_matrix()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # scipy warns at gimbal lock
                expected = ScipyRotation.from_matrix(matrix).as_euler("xyz")
            assert np.allclose(
                expected, Rotation.from_matrix(matrix).as_euler("xyz"), atol=TOL
            )


def test_rejects_the_same_input_scipy_rejects(rng):
    """Invalid input raises ValueError exactly where scipy raises it."""
    reflection = np.diag([1.0, 1.0, -1.0])
    degenerate = np.zeros((3, 3))
    for bad_matrix in (reflection, degenerate):
        with pytest.raises(ValueError):
            ScipyRotation.from_matrix(bad_matrix)
        with pytest.raises(ValueError):
            Rotation.from_matrix(bad_matrix)
    for seq, angles in (("", []), ("xyzx", [1, 2, 3, 4]), ("xx", [0.1, 0.2])):
        with pytest.raises(ValueError):
            ScipyRotation.from_euler(seq, angles)
        with pytest.raises(ValueError):
            Rotation.from_euler(seq, angles)


def test_rejects_what_it_does_not_support():
    """Unsupported input fails loudly rather than returning something wrong."""
    with pytest.raises(ValueError):
        Rotation.from_euler("XYZ", [0.1, 0.2, 0.3])  # intrinsic
    with pytest.raises(ValueError):
        Rotation.from_euler("xyz", [0.1, 0.2])  # wrong angle count
    with pytest.raises(ValueError):
        Rotation.from_matrix(np.eye(4))  # not 3x3
    with pytest.raises(ValueError):
        Rotation.from_matrix(np.zeros((2, 3, 3)))  # batched
    with pytest.raises(ValueError):
        Rotation.from_rotvec(np.zeros((2, 3)))  # batched
    with pytest.raises(ValueError):
        Rotation.from_matrix(np.eye(3)).as_euler("zyx")  # unsupported sequence
