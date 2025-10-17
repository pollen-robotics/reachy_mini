"""Protocol definitions for Reachy Mini client/server communication."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from reachy_mini.utils.interpolation import InterpolationTechnique


class GotoTaskRequest(BaseModel):
    """Class to represent a goto target task."""

    head: list[float] | None  # 4x4 flatten pose matrix
    antennas: list[float] | None  # [right_angle, left_angle] (in rads)
    duration: float
    method: InterpolationTechnique
    body_yaw: float | None


class PlayMoveTaskRequest(BaseModel):
    """Class to represent a play move task."""

    move_name: str


AnyTaskRequest = GotoTaskRequest | PlayMoveTaskRequest


class TaskRequest(BaseModel):
    """Class to represent any task request."""

    uuid: UUID
    req: AnyTaskRequest
    timestamp: datetime


class TaskProgress(BaseModel):
    """Class to represent task progress."""

    uuid: UUID
    finished: bool = False
    error: str | None = None
    timestamp: datetime
