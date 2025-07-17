"""Placo Kinematics for Reachy Mini.

This module provides the PlacoKinematics class for performing inverse and forward kinematics based on the Reachy Mini robot URDF using the Placo library.
"""

from typing import List, Optional

import numpy as np
import pinocchio as pin
import placo


class PlacoKinematics:
    """Placo Kinematics class for Reachy Mini.

    This class provides methods for inverse and forward kinematics using the Placo library and a URDF model of the Reachy Mini robot.
    """

    def __init__(
        self, urdf_path: str, dt: float = 0.02, automatic_body_yaw: bool = True
    ) -> None:
        """Initialize the PlacoKinematics class.

        Args:
            urdf_path (str): Path to the URDF file of the Reachy Mini robot.
            dt (float): Time step for the kinematics solver. Default is 0.02 seconds.
            automatic_body_yaw (bool): If True, the body yaw will be used to compute the IK and FK. Default is True.

        """
        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)

        self.ik_solver = placo.KinematicsSolver(self.robot)
        self.ik_solver.mask_fbase(True)

        self.fk_solver = placo.KinematicsSolver(self.robot)
        self.fk_solver.mask_fbase(True)

        self.automatic_body_yaw = automatic_body_yaw

        # they should be hard but we use soft to avoir singularities and
        # tradeoff the precision for the robustness
        constrant_type = "soft"  # "hard" or "soft"

        # IK closing tasks
        ik_closing_tasks = []
        for i in range(1, 6):
            ik_closing_task = self.ik_solver.add_relative_position_task(
                f"closing_{i}_1", f"closing_{i}_2", np.zeros(3)
            )
            ik_closing_task.configure(f"closing_{i}", constrant_type, 1.0)
            ik_closing_tasks.append(ik_closing_task)

        # FK closing tasks
        fk_closing_tasks = []
        for i in range(1, 6):
            fk_closing_task = self.fk_solver.add_relative_position_task(
                f"closing_{i}_1", f"closing_{i}_2", np.zeros(3)
            )
            fk_closing_task.configure(f"closing_{i}", constrant_type, 1.0)
            fk_closing_tasks.append(fk_closing_task)

        # Z offset for the head to make it easier to compute the IK and FK
        # This is the height of the head from the base of the robot
        self.head_z_offset = 0.177  # offset for the head height

        # IK head task
        self.head_starting_pose = np.eye(4)
        self.head_starting_pose[:3, 3][2] = self.head_z_offset
        self.head_frame = self.ik_solver.add_frame_task("head", self.head_starting_pose)
        self.head_frame.configure("head", "soft", 1.0, 1.0)

        self.head_frame.T_world_frame = self.head_starting_pose

        # regularization
        self.ik_yaw_joint_task = self.ik_solver.add_joints_task()
        self.ik_yaw_joint_task.set_joints({"all_yaw": 0})
        if self.automatic_body_yaw:
            self.ik_yaw_joint_task.configure("joints", "soft", 5e-5)
        else:
            self.ik_yaw_joint_task.configure("joints", "soft", 10.0)
            

        self.ik_joint_task = self.ik_solver.add_joints_task()
        self.ik_joint_task.set_joints({f"{k}": 0 for k in range(1, 7)})
        self.ik_joint_task.configure("joints", "soft", 1e-5)

        self.ik_solver.enable_velocity_limits(True)
        self.ik_solver.dt = dt

        # FK joint task
        self.head_joints_task = self.fk_solver.add_joints_task()
        self.head_joints_task.configure("joints", "soft", 1.0)

        # regularization
        self.fk_yaw_joint_task = self.fk_solver.add_joints_task()
        self.fk_yaw_joint_task.set_joints({"all_yaw": 0})
        self.fk_yaw_joint_task.configure("joints", "soft", 5e-5)

        self.fk_joint_task = self.fk_solver.add_joints_task()
        self.fk_joint_task.set_joints({f"{k}": 0 for k in range(1, 7)})
        self.fk_joint_task.configure("joints", "soft", 1e-5)

        # self.fk_solver.enable_velocity_limits(True)
        self.fk_solver.dt = dt

        # Actuated DoFs
        self.joints_names = [
            "all_yaw",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]

        # Passive DoFs to eliminate with constraint jacobian
        self.passive_joints_names = [
            "passive_1_x",
            "passive_1_y",
            "passive_2_x",
            "passive_2_y",
            "passive_3_x",
            "passive_3_y",
            "passive_4_x",
            "passive_4_y",
            "passive_5_x",
            "passive_5_y",
            "passive_6_x",
            "passive_6_y",
            "passive_7_x",
            "passive_7_y",
            "passive_7_z",
        ]

        # Retrieving indexes in the jacobian
        self.passives_idx = [
            self.robot.get_joint_v_offset(dof) for dof in self.passive_joints_names
        ]
        self.actives_idx = [
            self.robot.get_joint_v_offset(dof)
            for dof in self.robot.joint_names()
            if dof not in self.passive_joints_names
        ]
        self.actuated_idx = [
            self.robot.get_joint_v_offset(dof)
            for dof in self.robot.joint_names()
            if dof in self.joints_names
        ]

        # actuated dof indexes in active dofs
        self.actuated_idx_in_active = [
            i for i, idx in enumerate(self.actives_idx) if idx in self.actuated_idx
        ]

        # setup the collision model
        self.config_collision_model()

    def ik(
        self, pose: np.ndarray, body_yaw: float = None, check_collision: bool = False
    ) -> Optional[List[float]]:
        """Compute the inverse kinematics for the head for a given pose.

        Args:
            pose (np.ndarray): A 4x4 homogeneous transformation matrix
                representing the desired position and orientation of the head.
            body_yaw (float): Body yaw angle in radians.
            check_collision (bool): If True, checks for collisions after solving IK. (default: False)


        Returns:
            List[float]: A list of joint angles for the head.

        """
        _pose = pose.copy()

        if self.automatic_body_yaw: 
            self.ik_solver.unmask_dof("all_yaw")
        else:
            self.ik_solver.mask_dof("all_yaw")

        if body_yaw is not None:
            # TODO Is this how we set a new task goal?
            self.ik_yaw_joint_task.set_joints({"all_yaw": body_yaw})

        # set the head pose
        _pose[:3, 3][2] += self.head_z_offset  # offset the height of the head
        self.head_frame.T_world_frame = _pose

        q = self.robot.state.q.copy()
        for _ in range(10):
            try:
                self.ik_solver.solve(True)  # False to not update the kinematics
            except Exception as e:
                print(f"IK solver failed: {e}")
                self.robot.state.q = q
            self.robot.update_kinematics()

        # verify that there is no collision
        if check_collision and self.compute_collision():
            print("Collision detected, stopping ik...")
            self.robot.state.q = q  # revert to the previous state
            self.robot.update_kinematics()
            return None

        joints = []
        for joint_name in self.joints_names:
            joint = self.robot.get_joint(joint_name)
            joints.append(joint)
    
        if (not self.automatic_body_yaw) and (body_yaw is not None):
            # If body_yaw is set, we need to set the yaw joint to the desired value
            # because the IK solver will not control this DOF 
            joints[0] = body_yaw
            
        return joints

    def fk(
        self, joints_angles: List[float], check_collision=False
    ) -> Optional[np.ndarray]:
        """Compute the forward kinematics for the head given joint angles.

        Args:
            joints_angles (List[float]): A list of joint angles for the head.
            check_collision (bool): If True, checks for collisions after solving FK. (default: False)

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix

        """
        self.head_joints_task.set_joints(
            {
                "all_yaw": joints_angles[0],
                "1": joints_angles[1],
                "2": joints_angles[2],
                "3": joints_angles[3],
                "4": joints_angles[4],
                "5": joints_angles[5],
                "6": joints_angles[6],
            }
        )

        q = self.robot.state.q.copy()
        for _ in range(10):
            self.fk_solver.solve(True)
            self.robot.update_kinematics()

        if check_collision and self.compute_collision():
            print("Collision detected, stopping FK...")
            self.robot.state.q = q  # revert to the previous state
            return None

        T_world_head = self.robot.get_T_world_frame("head")
        T_world_head[:3, 3][2] -= self.head_z_offset  # offset the height of the head
        return T_world_head

    def config_collision_model(self):
        """Configure the collision model for the robot.

        Add collision pairs between the torso and the head colliders.
        """
        geom_model = self.robot.collision_model

        # name_torso_collider = "dc15_a01_case_b_dummy_10"
        # names_head_colliders = ["pp01063_stewart_plateform_7", "pp01063_stewart_plateform_11"]

        id_torso_collider = 12  # geom_model.getGeometryObjectId(name_torso_collider)
        id_head_colliders = [
            74,
            78,
        ]  # [geom_model.getGeometryObjectId(name) for name in names_head_colliders]

        for i in id_head_colliders:
            geom_model.addCollisionPair(
                pin.CollisionPair(id_torso_collider, i)
            )  # torso with head colliders

    def compute_collision(self, margin=0.005):
        """Compute the collision between the robot and the environment.

        Args:
            margin (float): The margin to consider for collision detection (default: 5mm).

        Returns:
            True if there is a collision, False otherwise.

        """
        collision_data = self.robot.collision_model.createData()
        data = self.robot.model.createData()

        # pin.computeCollisions(
        pin.computeDistances(
            self.robot.model,
            data,
            self.robot.collision_model,
            collision_data,
            self.robot.state.q,
        )

        # Iterate over all collision pairs
        for distance_result in collision_data.distanceResults:
            if distance_result.min_distance <= margin:
                return True  # Something is too close or colliding!

        return False  # Safe

    def compute_jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the Jacobian of the head frame with respect to the actuated DoFs.

        The jacobian in local world aligned.

        Args:
            q (np.ndarray, optional): Joint angles of the robot. If None, uses the current state of the robot. (default: None)

        Returns:
            np.ndarray: The Jacobian matrix.

        """
        # If q is provided, use it to compute the forward kinematics
        if q is not None:
            self.fk(q)

        # Computing the platform Jacobian
        # dx = Jp.dq
        Jp = self.robot.frame_jacobian("head", "local_world_aligned")

        # Computing the constraints Jacobian
        # 0 = Jc.dq
        constraints = []
        for i in range(1, 6):
            Jc = self.robot.relative_position_jacobian(
                f"closing_{i}_1", f"closing_{i}_2"
            )
            constraints.append(Jc)
        Jc = np.vstack(constraints)

        # Splitting jacobians as
        # Jp_a.dq_a + Jp_p.dq_p = dx
        Jp_a = Jp[:, self.actives_idx]
        Jp_p = Jp[:, self.passives_idx]
        # Jc_a.dq_a + Jc_p.dq_p = 0
        Jc_a = Jc[:, self.actives_idx]
        Jc_p = Jc[:, self.passives_idx]

        # Computing effector jacobian under constraints
        # Because constraint equation
        #       Jc_a.dq_a + Jc_p.dq_p = 0
        # can be written as:
        #       dq_p = - (Jc_p)^(⁻1) @ Jc_a @ dq_a
        # Then we can substitute dq_p in the first equation and get
        # This new jacobian
        J = Jp_a - Jp_p @ np.linalg.inv(Jc_p) @ Jc_a

        return J[:, self.actuated_idx_in_active]

    def compute_gravity_torque(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the gravity torque vector for the actuated joints of the robot.

        This method uses the static gravity compensation torques from the robot's dictionary.

        Args:
            q (np.ndarray, optional): Joint angles of the robot. If None, uses the current state of the robot. (default: None)

        Returns:
            np.ndarray: The gravity torque vector.

        """
        # If q is provided, use it to compute the forward kinematics
        if q is not None:
            self.fk(q)

        # Get the static gravity compensation torques for all joints
        # except the mobile base 6dofs
        grav_torque_all_joints = np.array(
            list(
                self.robot.static_gravity_compensation_torques_dict(
                    "pp01071_turning_bowl"
                ).values()
            )
        )

        # See the paper for more info (equations 4-9):
        #   https://hal.science/hal-03379538/file/BriotKhalil_SpringerEncyclRob_bookchapterPKMDyn.pdf#page=4
        #
        # Basically to compute the actuated torques necessary to compensate the gravity, we need to compute the
        # the equivalent wrench in the head frame that would be created if all the joints were actuated.
        #       wrench_eq = np.linalg.pinv(J_all_joints.T) @ torque_all_joints
        # And then we can compute the actuated torques as:
        #       torque_actuated = J_actuated.T @ wrench_eq
        J_all_joints = self.robot.frame_jacobian("head", "local_world_aligned")[
            :, 6:
        ]  # all joints except the mobile base 6dofs
        J_actuated = self.compute_jacobian()
        # using a single matrix G to compute the actuated torques
        G = J_actuated.T @ np.linalg.pinv(J_all_joints.T)

        # torques of actuated joints
        grav_torque_actuated = G @ grav_torque_all_joints

        # Compute the gravity torque
        return grav_torque_actuated
