"""Moved to ``reachy_mini.utils.daemon``; kept so existing imports keep working."""

from reachy_mini.utils.daemon import (
    CAMERA_PIPE_NAME,
    CAMERA_SOCKET_PATH,
    SimulationMode,
    daemon_check,
    find_serial_port,
    get_ip_address,
    is_local_camera_available,
    is_localhost,
)

__all__ = [
    "CAMERA_PIPE_NAME",
    "CAMERA_SOCKET_PATH",
    "SimulationMode",
    "daemon_check",
    "find_serial_port",
    "get_ip_address",
    "is_local_camera_available",
    "is_localhost",
]
