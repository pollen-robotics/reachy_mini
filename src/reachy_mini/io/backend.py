import threading
from typing import List


class Backend:
    """
    Base class for robot backends, simulated or real
    """

    def __init__(self) -> None:
        self.should_stop = threading.Event()
        self.ready = threading.Event()

        self.head_joint_positions = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.antenna_joint_positions = None  # [0, 1]
        self.joint_positions_publisher = None  # Placeholder for a publisher object

    def run(self):
        raise NotImplementedError("The method run should be overridden by subclasses.")

    def close(self) -> None:
        """
        Close the backend.
        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method close should be overridden by subclasses."
        )

    def set_joint_positions_publisher(self, publisher) -> None:
        self.joint_positions_publisher = publisher

    def set_head_joint_positions(self, positions: List[float]) -> None:
        self.head_joint_positions = positions

    def set_antenna_joint_positions(self, positions: List[float]) -> None:
        self.antenna_joint_positions = positions

    def set_torque(self, enabled: bool) -> None:
        raise NotImplementedError(
            "The method set_torque should be overridden by subclasses."
        )

    def get_head_joint_positions(self) -> List[float]:
        raise NotImplementedError(
            "The method get_head_joint_positions should be overridden by subclasses."
        )

    def get_antenna_joint_positions(self) -> List[float]:
        raise NotImplementedError(
            "The method get_antenna_joint_positions should be overridden by subclasses."
        )

    def get_status(self):
        """
        Returns backend statistics.
        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_status should be overridden by subclasses."
        )
