"""Mujoco utilities for Reachy Mini.

This module provides utility functions for working with MuJoCo models: joint
addresses and actuator names.
"""

from mujoco._structs import MjModel


def get_joint_addr_from_name(model: MjModel, name: str) -> int:
    """Return the address of a specified joint."""
    addr: int = model.joint(name).qposadr
    return addr


def get_actuator_names(model: MjModel) -> list[str]:
    """Return the list of the actuators names from the MuJoCo model."""
    actuator_names = [model.actuator(k).name for k in range(0, model.nu)]
    return actuator_names
