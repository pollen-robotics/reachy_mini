import numpy as np
import placo


class PlacoFK:
    def __init__(self, urdf_path):
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)

        closing_task_1 = self.solver.add_relative_position_task(
            "closing_1_1", "closing_1_2", np.zeros(3)
        )
        closing_task_1.configure("closing_1", "hard", 1.0)

        closing_task_2 = self.solver.add_relative_position_task(
            "closing_2_1", "closing_2_2", np.zeros(3)
        )
        closing_task_2.configure("closing_2", "hard", 1.0)

        closing_task_3 = self.solver.add_relative_position_task(
            "closing_3_1", "closing_3_2", np.zeros(3)
        )
        closing_task_3.configure("closing_3", "hard", 1.0)

        closing_task_4 = self.solver.add_relative_position_task(
            "closing_4_1", "closing_4_2", np.zeros(3)
        )
        closing_task_4.configure("closing_4", "hard", 1.0)

        closing_task_5 = self.solver.add_relative_position_task(
            "closing_5_1", "closing_5_2", np.zeros(3)
        )
        closing_task_5.configure("closing_5", "hard", 1.0)

        self.head_joints_task = self.solver.add_joints_task()
        self.head_joints_task.configure("joints", "soft", 1.0)

        # self.head_starting_pose = np.eye(4)
        # self.head_starting_pose[:3, 3][2] = 0.155
        # self.head_frame = self.solver.add_frame_task("head", self.head_starting_pose)
        # self.head_frame.configure("head", "soft", 1.0, 1.0)

        # self.head_frame.T_world_frame = self.head_starting_pose
        self.joints_names = ["1", "2", "3", "4", "5", "6"]

    def fk(self, joints_angles):
        self.head_joints_task.set_joints(
            {
                "1": joints_angles[0],
                "2": joints_angles[1],
                "3": joints_angles[2],
                "4": joints_angles[3],
                "5": joints_angles[4],
                "6": joints_angles[5],
            }
        )
        self.solver.solve(True)
        self.robot.update_kinematics()

        T_world_head = self.robot.get_T_world_frame("head")
        return T_world_head

if __name__ == "__main__":
    import os
    from pathlib import Path
    ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    placo_fk = PlacoFK(f"{ROOT_PATH}/descriptions/stewart_little_magnet/")
    print(np.around(placo_fk.fk([0, 0, 0, 0, 0, 0]), 3))
