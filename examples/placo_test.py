import os
import time
from pathlib import Path
import numpy as np
import pinocchio
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz
from scipy.spatial.transform import Rotation as R

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent


robot = placo.RobotWrapper(f"{ROOT_PATH}/descriptions/stewart_little_magnet")
solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)

closing_task_1 = solver.add_relative_position_task(
    "closing_1_1", "closing_1_2", np.zeros(3)
)
closing_task_1.configure("closing_1", "hard", 1.0)

closing_task_2 = solver.add_relative_position_task(
    "closing_2_1", "closing_2_2", np.zeros(3)
)
closing_task_2.configure("closing_2", "hard", 1.0)

closing_task_3 = solver.add_relative_position_task(
    "closing_3_1", "closing_3_2", np.zeros(3)
)
closing_task_3.configure("closing_3", "hard", 1.0)

closing_task_4 = solver.add_relative_position_task(
    "closing_4_1", "closing_4_2", np.zeros(3)
)
closing_task_4.configure("closing_4", "hard", 1.0)

closing_task_5 = solver.add_relative_position_task(
    "closing_5_1", "closing_5_2", np.zeros(3)
)
closing_task_5.configure("closing_5", "hard", 1.0)

head_starting_pose = np.eye(4)
head_starting_pose[:3, 3][2] = 0.155
head_frame = solver.add_frame_task("head", head_starting_pose)
head_frame.configure("head", "soft", 1.0, 1.0)

head_frame.T_world_frame = head_starting_pose

viz = robot_viz(robot)

start = time.time()
meshcat_decimation = 5
i = 0
while True:
    i += 1
    t = time.time() - start
    head_target = head_starting_pose.copy()
    head_target[:3, 3][0] += 0.01 * np.sin(2 * np.pi * 0.5 * t)

    # euler_rot = [
    #     0,
    #     0,
    #     # 0.2 * np.sin(2 * np.pi * 0.5 * t),
    #     0.5 * np.sin(2 * np.pi * 0.5 * t + np.pi),
    # ]
    # rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    # head_target[:3, :3] = rot_mat


    head_frame.T_world_frame = head_target

    solver.solve(True)
    robot.update_kinematics()

    if i % meshcat_decimation == 0:
        viz.display(robot.state.q)
        robot_frame_viz(robot, "head")
    time.sleep(0.01)
