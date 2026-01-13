"""Placo Kinematics for Reachy Mini.

This module provides the PlacoKinematics class for performing inverse and forward kinematics based on the Reachy Mini robot URDF using the Placo library.
"""

import logging
from typing import Annotated, List, Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin
import placo
from scipy.spatial.transform import Rotation as R


class PlacoKinematics:
    """Placo Kinematics class for Reachy Mini.

    This class provides methods for inverse and forward kinematics using the Placo library and a URDF model of the Reachy Mini robot.
    """

    def __init__(
        self,
        urdf_path: str,
        dt: float = 0.02,
        automatic_body_yaw: bool = False,
        check_collision: bool = False,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the PlacoKinematics class.

        Args:
            urdf_path (str): Path to the URDF file of the Reachy Mini robot.
            dt (float): Time step for the kinematics solver. Default is 0.02 seconds.
            automatic_body_yaw (bool): If True, the body yaw will be used to compute the IK and FK. Default is False.
            check_collision (bool): If True, checks for collisions after solving IK. (default: False)
            log_level (str): Logging level for the kinematics computations.

        """
        self.fk_reached_tol = np.deg2rad(
            0.1
        )  # 0.1 degrees tolerance for the FK reached condition

        if not urdf_path.endswith(".urdf"):
            urdf_path = f"{urdf_path}/{'robot_simple_collision.urdf' if check_collision else 'robot_no_collision.urdf'}"

        self.robot = placo.RobotWrapper(
            urdf_path, placo.Flags.collision_as_visual + placo.Flags.ignore_collisions
        )

        flags = (
            0
            if check_collision
            else placo.Flags.ignore_collisions + placo.Flags.collision_as_visual
        )
        self.robot_ik = placo.RobotWrapper(urdf_path, flags)

        self.ik_solver = placo.KinematicsSolver(self.robot_ik)
        self.ik_solver.mask_fbase(True)

        self.fk_solver = placo.KinematicsSolver(self.robot)
        self.fk_solver.mask_fbase(True)

        self.automatic_body_yaw = automatic_body_yaw
        self.check_collision = check_collision

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

        # we could go to soft limits to avoid over-constraining the IK
        # but the current implementation works robustly with hard limits
        # so we keep the hard limits for now
        constraint_type = "hard"  # "hard" or "soft"

        # IK closing tasks
        ik_closing_tasks = []
        for i in range(1, 6):
            ik_closing_task = self.ik_solver.add_relative_position_task(
                f"closing_{i}_1", f"closing_{i}_2", np.zeros(3)
            )
            ik_closing_task.configure(f"closing_{i}", constraint_type, 1.0)
            ik_closing_tasks.append(ik_closing_task)

        # FK closing tasks
        fk_closing_tasks = []
        for i in range(1, 6):
            fk_closing_task = self.fk_solver.add_relative_position_task(
                f"closing_{i}_1", f"closing_{i}_2", np.zeros(3)
            )
            fk_closing_task.configure(f"closing_{i}", constraint_type, 1.0)
            fk_closing_tasks.append(fk_closing_task)

        # Add the constraint between the rotated torso and the head
        # This will allow independent control of the torso and the head yaw
        # until this constraint is reached
        yaw_constraint = self.ik_solver.add_yaw_constraint(
            "dummy_torso_yaw", "head", np.deg2rad(55.0)
        )
        yaw_constraint.configure("rel_yaw", "hard")

        # Add the constraint to avoid the head from looking too far behind
        # Mostly due to some numerical problems 180 is always a bit tricky
        # Not really constraining because the this 180 pose is almost not
        # reachable with the real robot anyway
        yaw_constraint_abs = self.ik_solver.add_yaw_constraint(
            "body_foot_3dprint", "head", np.deg2rad(179.0)
        )
        yaw_constraint_abs.configure("abs_yaw", "hard")

        # Add a cone constraint for the head to not exceed a certain angle
        # This is to avoid the head from looking too far up or down
        self.fk_cone = self.ik_solver.add_cone_constraint(
            "body_foot_3dprint", "head", np.deg2rad(35.0)
        )
        self.fk_cone.configure("cone", "hard")
        self.fk_yaw_constraint = self.fk_solver.add_yaw_constraint(
            "dummy_torso_yaw", "head", np.deg2rad(55.0)
        )
        self.fk_yaw_constraint.configure("rel_yaw", "hard")

        # Add a cone constraint for the head to not exceed a certain angle
        # This is to avoid the head from looking too far up or down
        fk_cone = self.fk_solver.add_cone_constraint(
            "body_foot_3dprint", "head", np.deg2rad(35.0)
        )
        fk_cone.configure("cone", "hard")

        # Z offset for the head to make it easier to compute the IK and FK
        # This is the height of the head from the base of the robot
        self.head_z_offset = 0.177  # offset for the head height

        # IK head task
        self.head_starting_pose = np.eye(4)
        self.head_starting_pose[:3, 3][2] = self.head_z_offset
        self.head_frame = self.ik_solver.add_frame_task("head", self.head_starting_pose)
        # equivalance to ~1cm = 1deg weights
        # set to 5 to be higher than the 1.0 for the body yaw
        self.head_frame.configure(
            "head",
            "soft",
            5.0,  # in meters  # 1m
            5.0,  # in radians # 1rad
        )

        self.head_frame.T_world_frame = self.head_starting_pose

        # regularization
        self.ik_yaw_joint_task = self.ik_solver.add_joints_task()
        self.ik_yaw_joint_task.set_joints({"yaw_body": 0})
        if not self.automatic_body_yaw:
            self.ik_yaw_joint_task.configure("joints", "soft", 5e-5)
        else:
            self.ik_yaw_joint_task.configure("joints", "soft", 3.0)

        # joint limit tasks (values form URDF)
        self.ik_solver.enable_velocity_limits(True)
        self.ik_solver.enable_joint_limits(True)
        self.ik_solver.dt = dt

        # FK joint task
        self.head_joints_task = self.fk_solver.add_joints_task()
        self.head_joints_task.configure("joints", "soft", 5.0)
        # joint limit tasks (values form URDF)
        self.fk_solver.enable_velocity_limits(True)
        self.fk_solver.enable_joint_limits(True)
        self.fk_solver.dt = dt

        # Actuated DoFs
        self.joints_names = [
            "yaw_body",
            "stewart_1",
            "stewart_2",
            "stewart_3",
            "stewart_4",
            "stewart_5",
            "stewart_6",
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

        # set velocity limits to be artificially high
        # to enable faster convergence of the IK/FK solver
        max_vel = 13.0  # rad/s
        for joint_name in self.joints_names:
            if joint_name != "yaw_body":
                self.robot.set_velocity_limit(joint_name, max_vel)
                self.robot_ik.set_velocity_limit(joint_name, max_vel)

        self.robot.set_joint_limits("yaw_body", -2.8, 2.8)
        self.robot_ik.set_joint_limits("yaw_body", -2.8, 2.8)

        # initial state
        self._initial_q = self.robot.state.q.copy()
        self._initial_qd = np.zeros_like(self.robot.state.qd)
        self._initial_qdd = np.zeros_like(self.robot.state.qdd)

        # initial FK to set the head pose
        for _ in range(10):
            self.ik_solver.solve(True)  # False to not update the kinematics
            self.robot_ik.update_kinematics()

        # last good q to revert to in case of collision
        self._initial_q = self.robot_ik.state.q.copy()
        self._last_good_q = self.robot_ik.state.q.copy()

        # update the robot state to the initial state
        self._update_state_to_initial(self.robot)  # revert to the initial state
        self.robot.update_kinematics()

        if self.check_collision:
            ik_col = self.ik_solver.add_avoid_self_collisions_constraint()
            ik_col.self_collisions_margin = 0.001  # 1mm
            ik_col.self_collisions_trigger = 0.002  # 2mm
            ik_col.configure("avoid_self_collisions", "hard")

            # setup the collision model
            self.config_collision_model()

    def _update_state_to_initial(self, robot: placo.RobotWrapper) -> None:
        """Update the robot state to the initial state.

        It does not call update_kinematics, so the robot state is not updated.

        Args:
            robot (placo.RobotWrapper): The robot wrapper instance to update.

        """
        robot.state.q = self._initial_q
        robot.state.qd = self._initial_qd
        robot.state.qdd = self._initial_qdd

    def _pose_distance(
        self, pose1: npt.NDArray[np.float64], pose2: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        """Compute the orientation distance between two poses.

        Args:
            pose1 (np.ndarray): The first pose (4x4 homogeneous transformation matrix).
            pose2 (np.ndarray): The second pose (4x4 homogeneous transformation matrix).

        Returns:
            float: The Euler distance between the two poses.

        """
        euler1 = R.from_matrix(pose1[:3, :3]).as_euler("xyz")
        euler2 = R.from_matrix(pose2[:3, :3]).as_euler("xyz")
        p1 = pose1[:3, 3]
        p2 = pose2[:3, 3]
        return float(np.linalg.norm(euler1 - euler2)), float(np.linalg.norm(p1 - p2))

    def _closed_loop_constraints_valid(
        self, robot: placo.RobotWrapper, tol: float = 1e-2
    ) -> bool:
        """Check if all closed-loop constraints are satisfied.

        Args:
            robot (placo.RobotWrapper): The robot wrapper instance to check.
            tol (float): The tolerance for checking constraints (default: 1e-2).

        Returns:
            bool: True if all constraints are satisfied, False otherwise.

        """
        for i in range(1, 6):
            pos1 = robot.get_T_world_frame(f"closing_{i}_1")[:3, 3]
            pos2 = robot.get_T_world_frame(f"closing_{i}_2")[:3, 3]
            if not np.allclose(pos1, pos2, atol=tol):
                return False
        return True

    def _get_joint_values(self, robot: placo.RobotWrapper) -> List[float]:
        """Get the joint values from the robot state.

        Args:
            robot (placo.RobotWrapper): The robot wrapper instance to get joint values from.

        Returns:
            List[float]: A list of joint values.

        """
        joints = []
        for joint_name in self.joints_names:
            joint = robot.get_joint(joint_name)
            joints.append(joint)
        return joints

    def ik(
        self,
        pose: npt.NDArray[np.float64],
        body_yaw: float = 0.0,
        no_iterations: int = 2,
    ) -> Annotated[npt.NDArray[np.float64], (7,)] | None:
        """Compute the inverse kinematics for the head for a given pose.

        Args:
            pose (np.ndarray): A 4x4 homogeneous transformation matrix
                representing the desired position and orientation of the head.
            body_yaw (float): Body yaw angle in radians.
            no_iterations (int): Number of iterations to perform (default: 2). The higher the value, the more accurate the solution.

        Returns:
            List[float]: A list of joint angles for the head.

        """
        _pose = pose.copy()
        # set the head pose
        _pose[:3, 3][2] += self.head_z_offset  # offset the height of the head
        self.head_frame.T_world_frame = _pose
        # update the body_yaw task
        self.ik_yaw_joint_task.set_joints({"yaw_body": body_yaw})

        # check the starting configuration
        # if the poses are too far start from the initial configuration
        _dist_o, _dist_p = self._pose_distance(
            _pose, self.robot_ik.get_T_world_frame("head")
        )
        # if distance too small 0.1mm and 0.1 deg and the QP has converged (almost 0 velocity)
        _dist_by = np.abs(body_yaw - self.robot_ik.get_joint("yaw_body"))
        if (
            _dist_p < 0.1e-4
            and _dist_o < np.deg2rad(0.01)
            and _dist_by < np.deg2rad(0.01)
            and np.linalg.norm(self.robot_ik.state.qd) < 1e-4
        ):
            # no need to recalculate - return the current joint values
            return np.array(
                self._get_joint_values(self.robot_ik)
            )  # no need to solve IK
        if _dist_o >= np.pi:
            # distance too big between the current and the target pose
            # start the optim from zero position
            #
            # TO INVESTIGATE: Another way to do this would be not to start from 0 but
            # to set the target pose not to the actual target but to some intermediate pose
            self._update_state_to_initial(self.robot_ik)
            self.robot_ik.update_kinematics()
            self._logger.debug("IK: Poses too far, starting from initial configuration")

        done = True
        # do the initial ik
        for i in range(no_iterations):
            try:
                self.ik_solver.solve(True)  # False to not update the kinematics
            except Exception as e:
                self._logger.debug(f"IK solver failed: {e}, retrying...")
                done = False
                break
            self.robot_ik.update_kinematics()

        # if no problem in solving the IK check for constraint violation
        if done and (not self._closed_loop_constraints_valid(self.robot_ik)):
            self._logger.debug(
                "IK: Not all equality constraints are satisfied in IK, retrying..."
            )
            done = False

        # if there was an issue start from scratch
        if not done:
            # set the initial pose
            self._update_state_to_initial(self.robot_ik)
            self.robot_ik.update_kinematics()

            no_iterations += 2  # add a few more iterations
            # do the initial ik with 10 iterations
            for i in range(no_iterations):
                try:
                    self.ik_solver.solve(True)  # False to not update the kinematics
                except Exception as e:
                    self._logger.warning(f"IK solver failed: {e}, no solution found!")
                    return None
                self.robot_ik.update_kinematics()

        # Get the joint angles
        return np.array(self._get_joint_values(self.robot_ik))

    def fk(
        self,
        joints_angles: Annotated[npt.NDArray[np.float64], (7,)],
        no_iterations: int = 2,
    ) -> Optional[npt.NDArray[np.float64]]:
        """Compute the forward kinematics for the head given joint angles.

        Args:
            joints_angles (List[float]): A list of joint angles for the head.
            no_iterations (int): The number of iterations to use for the FK solver. (default: 2), the higher the more accurate the result.

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix

        """
        # check if we're already there
        _current_joints = self._get_joint_values(self.robot)
        # if the joint angles are the same (tol 1e-4) and teh QP has converged (max speed is 1e-4rad/s)
        # no need to compute the FK
        if (
            np.linalg.norm(np.array(_current_joints) - np.array(joints_angles))
            < self.fk_reached_tol
            and self.robot.state.qd.max() < 1e-4
        ):
            # no need to compute FK
            T_world_head: npt.NDArray[np.float64] = self.robot.get_T_world_frame("head")
            T_world_head[:3, 3][2] -= (
                self.head_z_offset
            )  # offset the height of the head
            return T_world_head

        # update the main task
        self.head_joints_task.set_joints(
            {
                "yaw_body": joints_angles[0],
                "stewart_1": joints_angles[1],
                "stewart_2": joints_angles[2],
                "stewart_3": joints_angles[3],
                "stewart_4": joints_angles[4],
                "stewart_5": joints_angles[5],
                "stewart_6": joints_angles[6],
            }
        )

        done = True
        # do the initial ik with 2 iterations
        for i in range(no_iterations):
            try:
                self.fk_solver.solve(True)  # False to not update the kinematics
            except Exception as e:
                self._logger.debug(f"FK solver failed: {e}, retrying...")
                done = False
                break
            self.robot.update_kinematics()

        if done and (not self._closed_loop_constraints_valid(self.robot)):
            self._logger.debug(
                "FK: Not all equality constraints are satisfied in FK, retrying..."
            )
            done = False

        if not done:
            self._update_state_to_initial(self.robot)  # revert to the previous state
            self.robot.update_kinematics()

            no_iterations += 2  # add a few more iterations
            # do the initial ik with 10 iterations
            for i in range(no_iterations):
                try:
                    self.fk_solver.solve(True)  # False to not update the kinematics
                except Exception as e:
                    self._logger.warning(f"FK solver failed: {e}, no solution found!")
                    return None
                self.robot.update_kinematics()

        # Get the head frame transformation
        T_world_head = self.robot.get_T_world_frame("head")
        T_world_head[:3, 3][2] -= self.head_z_offset  # offset the height of the head

        return T_world_head

    def config_collision_model(self) -> None:
        """Configure the collision model for the robot.

        Add collision pairs between the torso and the head colliders.
        """
        geom_model = self.robot_ik.collision_model

        id_torso_colliders = list(range(len(geom_model.geometryObjects) - 1))
        id_head_collider = len(geom_model.geometryObjects) - 1

        for i in id_torso_colliders:
            geom_model.addCollisionPair(
                pin.CollisionPair(id_head_collider, i)
            )  # torso with head colliders

    def compute_collision(self, margin: float = 0.005) -> bool:
        """Compute the collision between the robot and the environment.

        Args:
            margin (float): The margin to consider for collision detection (default: 5mm).

        Returns:
            True if there is a collision, False otherwise.

        """
        collision_data = self.robot_ik.collision_model.createData()
        data = self.robot_ik.model.createData()

        # pin.computeCollisions(
        pin.computeDistances(
            self.robot_ik.model,
            data,
            self.robot_ik.collision_model,
            collision_data,
            self.robot_ik.state.q,
        )

        # Iterate over all collision pairs
        for distance_result in collision_data.distanceResults:
            if distance_result.min_distance <= margin:
                return True  # Something is too close or colliding!

        return False  # Safe

    def compute_jacobian(
        self, q: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        """Compute the Jacobian of the head frame with respect to the actuated DoFs.

        The jacobian in local world aligned.

        Args:
            q (np.ndarray, optional): Joint angles of the robot. If None, uses the current state of the robot. (default: None)

        Returns:
            np.ndarray: The Jacobian matrix.

        """
        # If q is provided, use it to compute the forward kinematics
        if q is not None:
            self.fk(q, no_iterations=20)

        # Computing the platform Jacobian
        # dx = Jp.dq
        Jp: npt.NDArray[np.float64] = self.robot.frame_jacobian(
            "head", "local_world_aligned"
        )

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
        J: npt.NDArray[np.float64] = Jp_a - Jp_p @ np.linalg.inv(Jc_p) @ Jc_a

        return J[:, self.actuated_idx_in_active]

    def compute_gravity_torque(
        self, q: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
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
                    "body_foot_3dprint"
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
        J_all_joints: npt.NDArray[np.float64] = self.robot.frame_jacobian(
            "head", "local_world_aligned"
        )[:, 6:]  # all joints except the mobile base 6dofs
        J_actuated = self.compute_jacobian()
        # using a single matrix G to compute the actuated torques
        G = J_actuated.T @ np.linalg.pinv(J_all_joints.T)

        # torques of actuated joints
        grav_torque_actuated: npt.NDArray[np.float64] = G @ grav_torque_all_joints

        # Compute the gravity torque
        return grav_torque_actuated

    def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
        """Set the automatic body yaw.

        Args:
            automatic_body_yaw (bool): Whether to enable automatic body yaw.

        """
        self.automatic_body_yaw = automatic_body_yaw

        if not self.automatic_body_yaw:
            self.ik_yaw_joint_task.configure("joints", "soft", 3.0)
        else:
            self.ik_yaw_joint_task.configure("joints", "soft", 5e-5)

    def get_joint(self, joint_name: str) -> float:
        """Get the joint object by its name."""
        return float(self.robot.get_joint(joint_name))
