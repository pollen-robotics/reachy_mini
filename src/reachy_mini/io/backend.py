from typing import List


class Backend:
    def __init__(self):
        self.head_joint_positions = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.antenna_joint_positions = None  # [0, 1]
        self.torque_enabled: bool = False
        self.joint_positions_publisher = None  # Placeholder for a publisher object

    def set_joint_positions_publisher(self, publisher) -> None:
        self.joint_positions_publisher = publisher

    def set_head_joint_positions(self, positions: List[float]) -> None:
        self.head_joint_positions = positions

    def set_antenna_joint_positions(self, positions: List[float]) -> None:
        self.antenna_joint_positions = positions

    def set_torque(self, enabled: bool) -> None:
        self.torque_enabled = enabled

    def get_head_joint_positions(self) -> List[float]:
        """
        Returns head joints positions
        This method is a placeholder and should be overridden by subclasses.
        """

        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_antenna_joint_positions(self) -> List[float]:
        """
        Returns antenna joints positions
        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
