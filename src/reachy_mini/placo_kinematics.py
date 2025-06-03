import numpy as np
import placo


class PlacoKinematics:
    def __init__(self, urdf_path: str, dt: float = 0.02):
        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)

        self.ik_solver = placo.KinematicsSolver(self.robot)
        self.ik_solver.mask_fbase(True)

        self.fk_solver = placo.KinematicsSolver(self.robot)
        self.fk_solver.mask_fbase(True)

        # IK closing tasks
        ik_closing_task_1 = self.ik_solver.add_relative_position_task(
            "closing_1_1", "closing_1_2", np.zeros(3)
        )
        ik_closing_task_1.configure("closing_1", "hard", 1.0)

        ik_closing_task_2 = self.ik_solver.add_relative_position_task(
            "closing_2_1", "closing_2_2", np.zeros(3)
        )
        ik_closing_task_2.configure("closing_2", "hard", 1.0)

        ik_closing_task_3 = self.ik_solver.add_relative_position_task(
            "closing_3_1", "closing_3_2", np.zeros(3)
        )
        ik_closing_task_3.configure("closing_3", "hard", 1.0)

        ik_closing_task_4 = self.ik_solver.add_relative_position_task(
            "closing_4_1", "closing_4_2", np.zeros(3)
        )
        ik_closing_task_4.configure("closing_4", "hard", 1.0)

        ik_closing_task_5 = self.ik_solver.add_relative_position_task(
            "closing_5_1", "closing_5_2", np.zeros(3)
        )
        ik_closing_task_5.configure("closing_5", "hard", 1.0)

        # FK closing tasks
        fk_closing_task_1 = self.fk_solver.add_relative_position_task(
            "closing_1_1", "closing_1_2", np.zeros(3)
        )
        fk_closing_task_1.configure("closing_1", "hard", 1.0)

        fk_closing_task_2 = self.fk_solver.add_relative_position_task(
            "closing_2_1", "closing_2_2", np.zeros(3)
        )
        fk_closing_task_2.configure("closing_2", "hard", 1.0)

        fk_closing_task_3 = self.fk_solver.add_relative_position_task(
            "closing_3_1", "closing_3_2", np.zeros(3)
        )
        fk_closing_task_3.configure("closing_3", "hard", 1.0)

        fk_closing_task_4 = self.fk_solver.add_relative_position_task(
            "closing_4_1", "closing_4_2", np.zeros(3)
        )
        fk_closing_task_4.configure("closing_4", "hard", 1.0)

        fk_closing_task_5 = self.fk_solver.add_relative_position_task(
            "closing_5_1", "closing_5_2", np.zeros(3)
        )
        fk_closing_task_5.configure("closing_5", "hard", 1.0)

        self.joints_names = [
            "all_yaw",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "left_antenna",
            "right_antenna",
        ]

        # IK head task
        self.head_starting_pose = np.eye(4)
        self.head_starting_pose[:3, 3][2] = 0.177
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

        self.fk_solver.enable_velocity_limits(True)
        self.fk_solver.dt = dt

    def ik(self, pose):
        self.head_frame.T_world_frame = pose
        self.ik_solver.solve(True)
        self.robot.update_kinematics()

        joints = []
        for joint_name in self.joints_names:
            joint = self.robot.get_joint(joint_name)
            joints.append(joint)

        return joints

    def fk(self, joints_angles):
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
        self.fk_solver.solve(True)
        self.robot.update_kinematics()

        T_world_head = self.robot.get_T_world_frame("head")
        return T_world_head
