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


def fast_euler_from_matrix(R_mat: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Fast Euler angle extraction without creating Rotation object.
    
    For XYZ order (rotation matrix extrinsic):
    Faster than scipy.spatial.transform.Rotation.as_euler() which has overhead.
    """
    if order == "XYZ":
        # Direct extraction from rotation matrix for XYZ convention
        # Avoids Rotation object creation overhead (~2-3x faster)
        sin_y = -R_mat[2, 0]
        sin_y = np.clip(sin_y, -1.0, 1.0)  # Clamp to avoid numerical errors
        
        cos_y = np.sqrt(R_mat[0, 0]**2 + R_mat[1, 0]**2)
        y = np.arctan2(sin_y, cos_y)
        
        if np.abs(cos_y) > 1e-6:
            x = np.arctan2(R_mat[2, 1], R_mat[2, 2])
            z = np.arctan2(R_mat[1, 0], R_mat[0, 0])
        else:
            # Gimbal lock case
            x = 0.0
            z = np.arctan2(-R_mat[0, 1], R_mat[1, 1])
        
        return np.array([x, y, z])
    else:
        # Fall back to scipy for other orders
        return R.from_matrix(R_mat).as_euler(order)

# Duplicated for now.
SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# XL330 frame pose in the head frame
# exracted from URDF
T_HEAD_XL_330 = np.array([
    [ 0.4822, -0.7068, -0.5177,  0.0206],
    [ 0.1766, -0.5003,  0.8476, -0.0218],
    [-0.8581, -0.5001, -0.1164, -0.    ],
    [ 0.    ,  0.    ,  0.    ,  1.    ]])

# passive joint orientation offsets 
# extracted from URDF
PASSIVE_ORIENTATION_OFFSET = \
[
    [-0.13754,-0.0882156, 2.10349],    
    [-3.14159, 5.37396e-16, -3.14159],
    [0.373569, 0.0882156, -1.0381], 
    [-0.0860846, 0.0882156, 1.0381],
    [0.123977, 0.0882156, -1.0381],
    [3.0613, 0.0882156, 1.0381],
    [3.14159, 2.10388e-17, 4.15523e-17]
]

# stewart rod direction in passive frame
# exracted from URDF
STEWART_ROD_DIR_IN_PASSIVE_FRAME = np.array([
    [1, 0, 0], # rod 1
    [ 0.50606941, -0.85796418, -0.08826792],# rod 2
    [-1, 0, 0], # rod 3 
    [-1, 0, 0], # ....
    [-1, 0, 0],
    [-1, 0, 0]]
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
            data["motor_arm_length"], data["rod_length"]
        )

        self.start_body_yaw = 0.0

        self.motors = data["motors"]
        self.T_world_motor = []
        for motor in self.motors:
            T_w_motor = np.linalg.inv(motor["T_motor_world"])
            self.kin.add_branch(
                motor["branch_position"],
                T_w_motor,  # type: ignore[arg-type]
                1 if motor["solution"] else -1,
            )
            self.T_world_motor.append(T_w_motor)
            
        self.motor_arms_length = data["motor_arm_length"]
        self.rod_length = data["rod_length"]

        # TODO test with init head pose instead of sleep pose
        sleep_head_pose = SLEEP_HEAD_POSE.copy()
        sleep_head_pose[:3, 3][2] += self.head_z_offset
        self.kin.reset_forward_kinematics(sleep_head_pose)  # type: ignore[arg-type]

        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.WARNING)
        
        
        self.current_joints = np.zeros(7)   # inital joint angles
        self.passive_joints = np.zeros(21)  # inital passive joint angles
        # Cache passive correction rotations (computed once in __init__ would be even better)
        self.passive_corrections = [R.from_euler("xyz", offset).as_matrix() for offset in PASSIVE_ORIENTATION_OFFSET]
        

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
        while not ok:
            for _ in range(no_iterations):
                T_world_platform = np.array(
                    self.kin.forward_kinematics(_joint_angles, body_yaw)
                )
            assert T_world_platform is not None
            # Use faster Euler extraction (avoid Rotation object overhead)
            euler = fast_euler_from_matrix(T_world_platform[:3, :3], "XYZ")
            euler = np.degrees(euler)  # Convert to degrees if needed
            # check that head is upright. Recompute with epsilon adjustments if not
            if not (euler[0] > 90 or euler[0] < -90 or euler[1] > 90 or euler[1] < -90):
                ok = True
            else:
                self.logger.warning("Head is not upright, recomputing FK")
                body_yaw += 0.001
                _joint_angles = list(np.array(_joint_angles) + 0.001)
                tmp = np.eye(4)
                tmp[:3, 3][2] += self.head_z_offset
                self.kin.reset_forward_kinematics(tmp)  # type: ignore[arg-type]

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
        
        _pose = T_head.copy()
        _pose[:3, 3][2] += self.head_z_offset
    
        # Pre-allocate output array for better performance
        passive_joints = np.zeros(21)
    
        # Pre-compute common transforms to avoid redundant creation
        T_motor_servo_arm = np.eye(4)
        T_motor_servo_arm[:3, 3][0] = self.motor_arms_length
        
        # for each branch 
        #  - calculate the branch position on the platform in the world frame
        #  - calculate the servo arm joint position in the world frame
        #  - calculate the passive joint angles based on the two positions (roll, pitch, yaw)
        for i, motor in enumerate(self.motors):
            # Use direct position extraction instead of creating intermediate transforms
            branch_pos_world = _pose[:3, :3] @ motor["branch_position"] + _pose[:3, 3]
            
            # Compute servo rotation directly
            # rotating around Z axis
            cos_z = np.cos(joints[i+1])
            sin_z = np.sin(joints[i+1])
            R_servo = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            
            # Compute world servo arm position more efficiently
            T_world_motor = self.T_world_motor[i]
            servo_pos_local = R_servo @ T_motor_servo_arm[:3, 3]
            P_world_servo_arm = T_world_motor[:3, :3] @ servo_pos_local + T_world_motor[:3, 3]
            
            # Apply passive correction to orientation
            R_world_servo = T_world_motor[:3, :3] @ R_servo @ self.passive_corrections[i]
            
            # calculate the euler angles between the two world frame positions
            # imagining a straight line from the servo arm to the branch
            vec_servo_to_branch = branch_pos_world - P_world_servo_arm
            # Use direct matrix transpose instead of creating Rotation object for inverse
            vec_servo_to_branch_in_servo = R_world_servo.T @ vec_servo_to_branch
            
            # rod direction in the passive frame
            rod_dir = STEWART_ROD_DIR_IN_PASSIVE_FRAME[i]
            # should contain exactly the same value as the rod length
            norm_vec = np.linalg.norm(vec_servo_to_branch_in_servo)
            straight_line_dir = vec_servo_to_branch_in_servo / norm_vec
            R_servo_branch, _ = R.align_vectors(np.array([straight_line_dir]), np.array([rod_dir]))
            # Use fast Euler extraction instead of Rotation.as_euler()
            euler = fast_euler_from_matrix(R_servo_branch.as_matrix(), "XYZ")
            
            # Store directly in pre-allocated array instead of extend
            passive_joints[i*3:i*3+3] = euler
            
            if i == 5:
                # Compute 7th passive joint only for last branch
                # Calculate transformed position and rotation more efficiently
                R_servo_branch_mat = R_servo_branch.as_matrix()
                
                # Head XL330 target orientation
                R_head_xl330 = _pose[:3, :3] @ T_HEAD_XL_330[:3, :3]
                
                # Current rod orientation with correction
                R_rod_current = R_world_servo @ R_servo_branch_mat @ self.passive_corrections[6]
                
                # Compute relative rotation
                R_dof = R_rod_current.T @ R_head_xl330
                # Use fast Euler extraction
                euler_7 = fast_euler_from_matrix(R_dof, "XYZ")
                passive_joints[18:21] = euler_7
                
        self.passive_joints = passive_joints
        return self.passive_joints