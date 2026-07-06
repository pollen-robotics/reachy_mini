"""A goto move to a target head pose and/or antennas position."""

import numpy as np
import numpy.typing as npt

from reachy_mini.utils.interpolation import (
    InterpolationTechnique,
    linear_pose_interpolation,
    time_trajectory,
)

from .move import Move


class GotoMove(Move):
    """A goto move to a target head pose and/or antennas position."""

    def __init__(
        self,
        start_head_pose: npt.NDArray[np.float64],
        target_head_pose: npt.NDArray[np.float64] | None,
        start_antennas: npt.NDArray[np.float64],
        target_antennas: npt.NDArray[np.float64] | None,
        start_body_yaw: float,
        target_body_yaw: float | None,
        duration: float,
        method: InterpolationTechnique,
    ):
        """Set up the goto move."""
        self.start_head_pose = start_head_pose
        self.target_head_pose = (
            target_head_pose if target_head_pose is not None else start_head_pose
        )
        # The antenna encoders are multi-turn: a start captured from a raw
        # reading can sit whole turns away from the servo's single-turn
        # command space, and an interpolation crossing a +-180 deg boundary
        # teleports the physical target by a full turn (violent sweep seen
        # live 2026-07-06). Wrap the start into one turn; targets are
        # in-range by convention, so the straight path stays boundary-free.
        self.start_antennas = (
            np.mod(np.asarray(start_antennas, dtype=float) + np.pi, 2.0 * np.pi)
            - np.pi
        )
        self.target_antennas = (
            target_antennas if target_antennas is not None else self.start_antennas
        )
        self.start_body_yaw = start_body_yaw
        self.target_body_yaw = (
            target_body_yaw if target_body_yaw is not None else start_body_yaw
        )

        self._duration = duration
        self.method = method

    @property
    def duration(self) -> float:
        """Duration of the goto in seconds."""
        return self._duration

    def evaluate(
        self, t: float
    ) -> tuple[
        npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None
    ]:
        """Evaluate the goto at time t."""
        interp_time = time_trajectory(t / self.duration, method=self.method)

        interp_head_pose = linear_pose_interpolation(
            self.start_head_pose,
            self.target_head_pose,
            interp_time,
            yaw_as_scalar=True,
        )
        interp_antennas_joint = (
            self.start_antennas
            + (self.target_antennas - self.start_antennas) * interp_time
        )
        interp_body_yaw_joint = (
            self.start_body_yaw
            + (self.target_body_yaw - self.start_body_yaw) * interp_time
        )

        return interp_head_pose, interp_antennas_joint, interp_body_yaw_joint
