"""An analytical kinematics engine for Reachy Mini, using Rust bindings.

The inverse kinematics use an analytical method, while the forward kinematics
use a numerical method (Newton).
"""

import json
import logging
from importlib.resources import files
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from reachy_mini_rust_kinematics import ReachyMiniRustKinematics
from scipy.spatial.transform import Rotation as R

import reachy_mini
    

# Duplicated for now.
SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

class AnalyticalKinematics:
    """Reachy Mini Analytical Kinematics class, implemented in Rust with python bindings."""

    def __init__(self, automatic_body_yaw: bool = True) -> None:
        """Initialize."""
        assets_root_path: str = str(files(reachy_mini).joinpath("assets/"))
        data_path = assets_root_path + "/kinematics_data.json"
        data = json.load(open(data_path, "rb"))

        self.automatic_body_yaw = automatic_body_yaw

        self.head_z_offset = data["head_z_offset"]

        self.kin = ReachyMiniRustKinematics(
            json_file_path = data_path
        )

        self.start_body_yaw = 0.0

        # TODO test with init head pose instead of sleep pose
        sleep_head_pose = SLEEP_HEAD_POSE.copy()
        sleep_head_pose[:3, 3][2] += self.head_z_offset
        self.kin.reset_forward_kinematics(sleep_head_pose)  # type: ignore[arg-type]

        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.WARNING)
        self.current_joints = np.zeros(7)   # inital joint angles
        self.passive_joints = np.zeros(21)  # inital passive joint angles

    def ik(
        self,
        pose: Annotated[NDArray[np.float64], (4, 4)],
        body_yaw: float = 0.0,
        check_collision: bool = False,
        no_iterations: int = 0,
    ) -> Annotated[NDArray[np.float64], (7,)]:
        """Compute the inverse kinematics for a given head pose.

        check_collision and no_iterations are not used by AnalyticalKinematics. We keep them for compatibility with the other kinematics engines
        """
        _pose = pose.copy()
        _pose[:3, 3][2] += self.head_z_offset

        reachy_joints = []
        if self.automatic_body_yaw:
            # inverse kinematics solution that modulates the body yaw to
            # stay within the mechanical limits (max_body_yaw)
            # additionally it makes sure the the relative yaw between the body and the head
            # stays within the mechanical limits (max_relative_yaw)
            reachy_joints = self.kin.inverse_kinematics_safe(
                _pose,  # type: ignore[arg-type]
                body_yaw=body_yaw,
                max_relative_yaw=np.deg2rad(65),
                max_body_yaw=np.deg2rad(160),
            )
        else:
            # direct inverse kinematics solution with given body yaw
            # it does not modify the body yaw
            stewart_joints = self.kin.inverse_kinematics(_pose, body_yaw)  # type: ignore[arg-type]
            reachy_joints = [body_yaw] + stewart_joints
            
        self.current_joints
        return np.array(reachy_joints)

    def fk(
        self,
        joint_angles: Annotated[NDArray[np.float64], (7,)],
        check_collision: bool = False,
        no_iterations: int = 3,
    ) -> Annotated[NDArray[np.float64], (4, 4)]:
        """Compute the forward kinematics for a given set of joint angles.

        check_collision is not used by AnalyticalKinematics.
        """
        body_yaw = joint_angles[0]
        self.current_joints = np.array(joint_angles)

        _joint_angles = joint_angles[1:].tolist()

        if no_iterations < 1:
            raise ValueError("no_iterations must be at least 1")

        T_world_platform = None
        ok = False
        no_tries = 0
        while not ok and no_tries < 5:
            no_tries += 1
            for _ in range(no_iterations):
                T_world_platform = np.array(
                    self.kin.forward_kinematics(_joint_angles, body_yaw)
                )
            assert T_world_platform is not None
            # check if Z axis if too low 
            if T_world_platform[2, 3]  < 0.1: 
                self.logger.warning(f"WARNING FK: Head Z position is below 0, recomputing FK, try no. {no_tries}")
                body_yaw += 0.001
                _joint_angles = list(np.array(_joint_angles) + 0.001)
                tmp = np.eye(4)
                #tmp[:3,:3] += np.random.randn(3,3)*1e-3
                tmp[:3, 3][2] += self.head_z_offset
                self.kin.reset_forward_kinematics(tmp)  # type: ignore[arg-type]
                continue
            # Use faster Euler extraction (avoid Rotation object overhead)
            euler = R.from_matrix(T_world_platform[:3, :3]).as_euler(
                "xyz", degrees=True
            )
            # check that head is upright. Recompute with epsilon adjustments if not
            body_yaw_deg = np.degrees(body_yaw)
            if not (euler[0] > 90 or euler[0] < -90 or euler[1] > 90 or euler[1] < -90 or abs(euler[2] - body_yaw_deg) > 90):
                ok = True
            else:
                if (euler[0] > 90 or euler[0] < -90 or euler[1] > 90 or euler[1] < -90):
                    self.logger.warning(f"WARNING FK: Head is not upright, recomputing FK, try no. {no_tries}")
                elif abs(euler[2] - body_yaw_deg) > 90:
                    self.logger.warning(f"WARNING FK: Head yaw relative to body yaw is too large, recomputing FK, try no. {no_tries}")
                body_yaw += 0.001
                _joint_angles = list(np.array(_joint_angles) + 0.001)
                tmp = np.eye(4)
                #tmp[:3,:3] += np.random.randn(3,3)*1e-3
                tmp[:3, 3][2] += self.head_z_offset
                self.kin.reset_forward_kinematics(tmp)  # type: ignore[arg-type]

        if not ok:
            self.logger.error("ERROR FK: Could not compute a valid FK solution after maximum iterations")
            return None
        
        assert T_world_platform is not None

        T_world_platform[:3, 3][2] -= self.head_z_offset
        
        return T_world_platform

    def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
        """Set the automatic body yaw.

        Args:
            automatic_body_yaw (bool): Whether to enable automatic body yaw.

        """
        self.automatic_body_yaw = automatic_body_yaw
        
    def get_joint(self, joint_name: str) -> float:
        """Get the joint object by its name."""
        if "yaw_body" == joint_name:
            return self.current_joints[0]

        if "stewart_" in joint_name:
            index = int(joint_name.split("_")[1]) -1
            return self.current_joints[index + 1]

        # the structueal passive joints
        # 1_x, 1_y, 1_z, 2_x, 2_y, 2_z, ..., 7_x, 7_y, 7_z
        if "passive_" in joint_name:
            index = int(joint_name.split("_")[1]) -1
            axis = joint_name.split("_")[2]
            axis_map = {"x": 0, "y": 1, "z": 2}
            return self.passive_joints[index * 3 + axis_map[axis]]


        return 0.0
    
    def calculate_passive_joints(self, joints: np.ndarray = None, T_head: np.ndarray = None) -> np.ndarray:
        """Calculate the passive joint angles based on the current joint angles."""
        
        if joints is None:
            joints = self.current_joints
        if T_head is None:
            T_head = self.fk(joints, no_iterations=10)
            
        self.passive_joints = self.kin.calculate_passive_joints(joints, T_head)
        return self.passive_joints