"""WebRTC module for Reachy Mini daemon.

Provides WebRTC DataChannel-based communication for controlling the robot
from HTTPS frontends without mixed-content issues.
"""

from reachy_mini.daemon.webrtc.peer import WebRTCPeer
from reachy_mini.daemon.webrtc.manager import ConnectionManager

__all__ = ["WebRTCPeer", "ConnectionManager"]
