"""Utilities for managing relative motion timeouts and smooth decay."""

import time
from typing import List, Tuple

import numpy as np

from reachy_mini.utils.interpolation import linear_pose_interpolation


class RelativeOffsetManager:
    """Manages relative offsets with timeout and smooth decay to zero."""
    
    def __init__(self, timeout_seconds: float = 1.0, decay_duration: float = 1.0):
        """Initialize the relative offset manager.
        
        Args:
            timeout_seconds: Time after which offsets start decaying if not updated
            decay_duration: Duration over which offsets smoothly decay to zero
        """
        self.timeout_seconds = timeout_seconds
        self.decay_duration = decay_duration
        
        # Tracking when offsets were last updated
        self.last_relative_command_time = 0.0
        
        # Current offsets
        self.head_pose_offset = np.eye(4)
        self.body_yaw_offset = 0.0
        self.antenna_joint_positions_offset = [0.0, 0.0]
        
        # Stored offsets at start of decay (for smooth interpolation)
        self._decay_start_time = None
        self._decay_start_head_pose = None
        self._decay_start_body_yaw = None
        self._decay_start_antenna_offsets = None
        
    def update_offsets(self, head_pose_offset: np.ndarray = None, 
                      body_yaw_offset: float = None,
                      antenna_offsets: List[float] = None) -> None:
        """Update relative offsets and reset timeout."""
        current_time = time.time()
        self.last_relative_command_time = current_time
        
        # Reset decay state since we got new commands
        self._decay_start_time = None
        
        # Update provided offsets
        if head_pose_offset is not None:
            self.head_pose_offset = head_pose_offset
        if body_yaw_offset is not None:
            self.body_yaw_offset = body_yaw_offset
        if antenna_offsets is not None:
            self.antenna_joint_positions_offset = antenna_offsets[:]
    
    def get_current_offsets(self) -> Tuple[np.ndarray, float, List[float]]:
        """Get current offsets, applying smooth decay if timeout has occurred."""
        current_time = time.time()
        time_since_last_command = current_time - self.last_relative_command_time
        
        # No timeout yet - return current offsets
        if time_since_last_command <= self.timeout_seconds:
            return self.head_pose_offset, self.body_yaw_offset, self.antenna_joint_positions_offset
        
        # Start decay if we haven't already
        if self._decay_start_time is None:
            self._decay_start_time = self.last_relative_command_time + self.timeout_seconds
            self._decay_start_head_pose = self.head_pose_offset.copy()
            self._decay_start_body_yaw = self.body_yaw_offset
            self._decay_start_antenna_offsets = self.antenna_joint_positions_offset[:]
        
        # Calculate decay progress
        decay_elapsed = current_time - self._decay_start_time
        if decay_elapsed >= self.decay_duration:
            # Decay complete - return zeros
            self.head_pose_offset = np.eye(4)
            self.body_yaw_offset = 0.0
            self.antenna_joint_positions_offset = [0.0, 0.0]
            return self.head_pose_offset, self.body_yaw_offset, self.antenna_joint_positions_offset
        
        # Smooth interpolation from start values to zeros
        t = decay_elapsed / self.decay_duration  # 0 to 1
        
        # Head pose: interpolate from decay_start_pose to identity
        identity_pose = np.eye(4)
        current_head_pose = linear_pose_interpolation(
            self._decay_start_head_pose, identity_pose, t
        )
        
        # Body yaw: linear interpolation to 0
        current_body_yaw = self._decay_start_body_yaw * (1.0 - t)
        
        # Antenna offsets: linear interpolation to [0, 0]
        current_antenna_offsets = [
            offset * (1.0 - t) for offset in self._decay_start_antenna_offsets
        ]
        
        # Update stored values for next call
        self.head_pose_offset = current_head_pose
        self.body_yaw_offset = current_body_yaw
        self.antenna_joint_positions_offset = current_antenna_offsets
        
        return self.head_pose_offset, self.body_yaw_offset, self.antenna_joint_positions_offset
    
    def is_in_timeout_decay(self) -> bool:
        """Check if offsets are currently decaying due to timeout."""
        if self._decay_start_time is None:
            return False
        current_time = time.time()
        decay_elapsed = current_time - self._decay_start_time
        return 0 <= decay_elapsed < self.decay_duration
    
    def force_reset(self) -> None:
        """Immediately reset all offsets to zero."""
        self.head_pose_offset = np.eye(4)
        self.body_yaw_offset = 0.0
        self.antenna_joint_positions_offset = [0.0, 0.0]
        self._decay_start_time = None