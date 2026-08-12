"""Minimal 3D rotation type, a drop-in for the slice of scipy we actually use.

Importing ``scipy.spatial.transform.Rotation`` costs about 1.7 s of cold
import on the wireless robot's CM4, and almost none of it is rotation maths:
``scipy.spatial`` eagerly pulls in the KD-tree, ``scipy.sparse``, the
array-API compatibility layer and ``numpy.f2py`` (a Fortran source parser) on
the way. The SDK only ever needs eight operations on a single 3x3 matrix, so
they live here instead, in plain numpy.

Conventions match ``scipy.spatial.transform.Rotation`` exactly:

* a lowercase sequence (``"xyz"``, ``"z"``) means *extrinsic* rotations about
  fixed axes, so ``from_euler("xyz", [a, b, c])`` is ``Rz(c) @ Ry(b) @ Rx(a)``;
* ``as_euler("xyz")`` returns ``[a, b, c]`` with ``b`` in ``[-pi/2, pi/2]`` and
  ``a``, ``c`` in ``[-pi, pi]``, pinning the third angle to 0 in gimbal lock;
* ``as_rotvec`` returns the canonical vector, so its norm is at most ``pi``.

``tests/unit_tests/test_rotation_matches_scipy.py`` fuzzes every operation
against scipy itself, which stays a dependency purely as that oracle.

Only the API shape and conventions are borrowed from scipy (BSD-3-Clause);
the implementation is independent, built from the standard published
algorithms (Rodrigues' formula, Shepperd's quaternion extraction, SVD
orthonormalisation).

Only single rotations are supported, which is all the SDK constructs. scipy is
built around batches and beats this implementation by a wide margin on arrays;
it is only slower on the one-rotation-at-a-time calls the SDK actually makes,
where its per-call validation and dispatch dominate the arithmetic.
"""

import numpy as np
import numpy.typing as npt

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_matrix(axis: str, angle: float) -> npt.NDArray[np.float64]:
    """Build the rotation matrix for a single named axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    if axis == "y":
        return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    if axis == "z":
        return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"unknown rotation axis {axis!r}, expected one of 'xyz'")


def _skew(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the cross-product matrix of a 3-vector."""
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class Rotation:
    """A single 3D rotation, stored as a 3x3 matrix."""

    __slots__ = ("_matrix",)

    def __init__(self, matrix: npt.NDArray[np.float64]) -> None:
        """Wrap an already-valid rotation matrix. Prefer the from_* constructors."""
        self._matrix = matrix

    # --- constructors ---------------------------------------------------

    @classmethod
    def from_matrix(cls, matrix: npt.ArrayLike) -> "Rotation":
        """Build a rotation from a 3x3 matrix, orthonormalising it like scipy does.

        Args:
            matrix: A 3x3 array close to a rotation matrix.

        Returns:
            The nearest rotation, so downstream ``as_euler`` / ``as_rotvec``
            stay well defined on matrices that have drifted.

        """
        array = np.asarray(matrix, dtype=float)
        if array.shape != (3, 3):
            raise ValueError(f"expected a single 3x3 matrix, got shape {array.shape}")
        if np.linalg.det(array) <= 0:
            # scipy rejects these too: a reflection or a degenerate matrix is
            # corrupt input, and silently "repairing" it would hide the bug.
            raise ValueError(
                "Non-positive determinant (left-handed or null coordinate "
                "frame) in rotation matrix"
            )
        u, _, vt = np.linalg.svd(array)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        return cls(rotation)

    @classmethod
    def from_euler(
        cls, seq: str, angles: npt.ArrayLike, degrees: bool = False
    ) -> "Rotation":
        """Build a rotation from extrinsic Euler angles.

        Args:
            seq: Axis sequence of 1 to 3 lowercase characters from ``"xyz"``.
                Lowercase means extrinsic, matching scipy.
            angles: One angle per axis in ``seq``. A scalar is accepted for a
                single-axis sequence.
            degrees: Whether the angles are in degrees rather than radians.

        Returns:
            The composed rotation.

        """
        values = np.atleast_1d(np.asarray(angles, dtype=float))
        if degrees:
            values = np.deg2rad(values)
        # Same rejections as scipy, in the same order.
        if not 1 <= len(seq) <= 3:
            raise ValueError(
                "Expected axis specification to be a non-empty string of "
                f"upto 3 characters, got {seq!r}"
            )
        if seq != seq.lower():
            raise ValueError(
                f"sequence {seq!r} is intrinsic; only extrinsic (lowercase) "
                "sequences are supported"
            )
        if any(axis not in _AXIS_INDEX for axis in seq):
            raise ValueError(f"sequence {seq!r} must only contain 'x', 'y' or 'z'")
        if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
            raise ValueError(f"Expected consecutive axes to be different, got {seq}")
        if len(seq) != values.size:
            raise ValueError(
                f"sequence {seq!r} needs {len(seq)} angle(s), got {values.size}"
            )

        matrix = np.eye(3)
        # Extrinsic: each later axis rotates about a fixed frame, so it
        # multiplies on the left.
        for axis, angle in zip(seq, values):
            matrix = _axis_matrix(axis, float(angle)) @ matrix
        return cls(matrix)

    @classmethod
    def from_rotvec(cls, rotvec: npt.ArrayLike, degrees: bool = False) -> "Rotation":
        """Build a rotation from a rotation vector (Rodrigues' formula).

        Args:
            rotvec: A 3-vector whose direction is the axis and whose norm is
                the angle.
            degrees: Whether the norm is in degrees rather than radians.

        Returns:
            The corresponding rotation.

        """
        vector = np.asarray(rotvec, dtype=float)
        if vector.shape != (3,):
            raise ValueError(f"expected a single 3-vector, got shape {vector.shape}")
        if degrees:
            vector = np.deg2rad(vector)

        theta = float(np.linalg.norm(vector))
        if theta < 1e-12:
            # Second-order expansion, exact to float precision for tiny angles
            # and free of the 0/0 in the axis normalisation.
            skew = _skew(vector)
            return cls(np.eye(3) + skew + 0.5 * (skew @ skew))
        skew = _skew(vector / theta)
        return cls(
            np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
        )

    # --- accessors ------------------------------------------------------

    def as_matrix(self) -> npt.NDArray[np.float64]:
        """Return the rotation as a fresh 3x3 matrix."""
        return self._matrix.copy()

    def as_euler(self, seq: str, degrees: bool = False) -> npt.NDArray[np.float64]:
        """Return the extrinsic Euler angles for the given sequence.

        Args:
            seq: Only ``"xyz"`` is supported, the sole sequence the SDK uses.
            degrees: Whether to return degrees rather than radians.

        Returns:
            The three angles, with the middle one in ``[-pi/2, pi/2]``.

        """
        if seq != "xyz":
            raise ValueError(
                f"sequence {seq!r} is not supported, only the extrinsic 'xyz'"
            )
        matrix = self._matrix
        sin_beta = float(np.clip(-matrix[2, 0], -1.0, 1.0))
        beta = np.arcsin(sin_beta)
        cos_beta = np.sqrt(max(0.0, 1.0 - sin_beta * sin_beta))

        if cos_beta > 1e-7:
            alpha = np.arctan2(matrix[2, 1], matrix[2, 2])
            gamma = np.arctan2(matrix[1, 0], matrix[0, 0])
        else:
            # Gimbal lock: only alpha -/+ gamma is determined, so pin gamma to
            # 0 and fold the whole rotation into alpha, as scipy does.
            gamma = 0.0
            if sin_beta > 0:
                alpha = np.arctan2(matrix[0, 1], matrix[1, 1])
            else:
                alpha = np.arctan2(-matrix[0, 1], matrix[1, 1])

        angles = np.array([alpha, beta, gamma])
        return np.rad2deg(angles) if degrees else angles

    def as_rotvec(self, degrees: bool = False) -> npt.NDArray[np.float64]:
        """Return the rotation vector, with a norm of at most pi.

        Args:
            degrees: Whether the norm should be in degrees rather than radians.

        Returns:
            The axis scaled by the angle.

        """
        matrix = self._matrix
        # Via a quaternion (Shepperd's method), which stays well conditioned
        # near pi where the naive trace formula loses all of its digits.
        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2
            w = 0.25 * scale
            xyz = (
                np.array(
                    [
                        matrix[2, 1] - matrix[1, 2],
                        matrix[0, 2] - matrix[2, 0],
                        matrix[1, 0] - matrix[0, 1],
                    ]
                )
                / scale
            )
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            xyz = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif matrix[1, 1] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            xyz = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            xyz = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )

        if w < 0:  # canonical hemisphere, so the angle stays within [0, pi]
            xyz, w = -xyz, -w
        norm = float(np.linalg.norm(xyz))
        if norm < 1e-12:
            rotvec = 2.0 * xyz
        else:
            rotvec = xyz * (2.0 * float(np.arctan2(norm, w)) / norm)
        out: npt.NDArray[np.float64] = np.rad2deg(rotvec) if degrees else rotvec
        return out

    # --- algebra --------------------------------------------------------

    def __mul__(self, other: "Rotation") -> "Rotation":
        """Compose two rotations, applying ``other`` first."""
        if not isinstance(other, Rotation):
            return NotImplemented
        return Rotation(self._matrix @ other._matrix)

    def inv(self) -> "Rotation":
        """Return the inverse rotation."""
        return Rotation(self._matrix.T)

    def __repr__(self) -> str:
        """Return a short, readable representation."""
        roll, pitch, yaw = self.as_euler("xyz", degrees=True)
        return f"Rotation(rpy_deg=[{roll:.3f}, {pitch:.3f}, {yaw:.3f}])"
