import numpy as np
import placo
from typing import List
import pinocchio as pin


class PlacoKinematics:
    def __init__(self, urdf_path: str, dt: float = 0.02) -> None:
        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)

        self.ik_solver = placo.KinematicsSolver(self.robot)
        self.ik_solver.mask_fbase(True)

        self.fk_solver = placo.KinematicsSolver(self.robot)
        self.fk_solver.mask_fbase(True)

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

        self.joints_names = [
            "all_yaw",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]

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
        self.ik_yaw_joint_task.configure("joints", "soft", 5e-5)

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

        # setup the collision model
        self.config_collision_model()

    def ik(self, pose: np.ndarray, check_collision: bool = False) -> List[float]:
        """
        Computes the inverse kinematics for the head for a given pose.

        Args:
            pose (np.ndarray): A 4x4 homogeneous transformation matrix
                representing the desired position and orientation of the head.
            check_collision (bool): If True, checks for collisions after solving IK. (default: False)
            

        Returns:
            List[float]: A list of joint angles for the head.
        """

        _pose = pose.copy()
        
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

        return joints

    def fk(self, joints_angles: List[float], check_collision=False) -> np.ndarray:
        """
        Computes the forward kinematics for the head given joint angles.

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

    def compute_collision(self):
        """
        Compute the collision between the robot and the environment.
        :return: True if there is a collision, False otherwise.
        """
        collision_data = self.robot.collision_model.createData()
        data = self.robot.model.createData()
        # pin.forwardKinematics(model, data, self.robot.state.q)
        # pin.updateFramePlacements(model, data)
        return pin.computeCollisions(
            self.robot.model,
            data,
            self.robot.collision_model,
            collision_data,
            self.robot.state.q,
        )
