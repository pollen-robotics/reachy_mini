import threading
from typing import Dict, List, Optional, Tuple, Union


class Backend:
    def __init__(self):
        self.should_stop = threading.Event()

        self.head_joint_positions = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.antenna_joint_positions = None  # [0, 1]
        self.joint_positions_publisher = None  # Placeholder for a publisher object
        self.led = None

    def set_joint_positions_publisher(self, publisher) -> None:
        self.joint_positions_publisher = publisher

    def set_head_joint_positions(self, positions: List[float]) -> None:
        self.head_joint_positions = positions

    def set_antenna_joint_positions(self, positions: List[float]) -> None:
        self.antenna_joint_positions = positions

    def set_torque(self, enabled: bool) -> None:
        raise NotImplementedError("This method should be overridden by subclasses.")

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

    def set_led_colors(
        self,
        colors: Union[
            List[Optional[Tuple[int, int, int]]], Dict[int, Tuple[int, int, int]]
        ],
        duration: Optional[float] = None,
    ):
        """
        Set the colors of the LEDs.
        This method is a placeholder and should be overridden by subclasses.
        """

    def clear_led(self):
        """
        Clear the LED colors.
        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def close(self):
        """
        Close the backend connection and stop any threads.
        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
