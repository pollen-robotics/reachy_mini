from math import atan, sqrt
import numpy as np
import time
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz
from placo_utils.tf import tf
from .placo_kinematics import PlacoKinematics

urdf_path = "descriptions/reachy_mini/urdf/robot.urdf"
solver = PlacoKinematics(urdf_path, 0.02)
robot = solver.robot
robot.update_kinematics()

viz = robot_viz(robot)

motors = [
    {"name": "1", "branch_frame": "closing_1_2", "offset": np.pi, "solution": 1},
    {"name": "2", "branch_frame": "closing_2_2", "offset": 0, "solution": 0},
    {"name": "3", "branch_frame": "closing_3_2", "offset": 0, "solution": 1},
    {"name": "4", "branch_frame": "closing_4_2", "offset": 0, "solution": 0},
    {"name": "5", "branch_frame": "closing_5_2", "offset": np.pi, "solution": 1},
    {"name": "6", "branch_frame": "passive_7_link_y", "offset": 0, "solution": 0},
]

# Measuring lengths for the arm and branch (constants could be used)
T_world_head = robot.get_T_world_frame("head")
T_world_1 = robot.get_T_world_frame("1")
T_world_arm1 = robot.get_T_world_frame("passive_1_link_x")

T_1_arm1 = np.linalg.inv(T_world_1) @ T_world_arm1
arm_z = T_1_arm1[2, 3]
servo_arm_length = np.linalg.norm(T_1_arm1[:2, 3])
T_world_branch1 = robot.get_T_world_frame("closing_1_2")
T_arm1_branch1 = np.linalg.inv(T_world_arm1) @ T_world_branch1
branch_length = np.linalg.norm(T_arm1_branch1[:3, 3])

# Finding the 6 branches position in the head frame, and building T_world_motor frame
# T_world_motor is a frame where the motor is located, correcting for the arm_z offset
for motor in motors:
    T_world_branch = robot.get_T_world_frame(motor["branch_frame"])
    T_head_branch = np.linalg.inv(T_world_head) @ T_world_branch
    T_world_motor = robot.get_T_world_frame(motor["name"]) @ tf.translation_matrix(
        (0, 0, arm_z)
    )
    motor["T_world_motor"] = T_world_motor
    motor["branch_position"] = T_head_branch[:3, 3]

# See compute_analytical_kinematics.py 
def ik_branch(px, py, pz):
    rs = servo_arm_length
    rp = branch_length

    return [
        2
        * atan(
            (
                2 * py * rs
                - sqrt(
                    -(px**4)
                    - 2 * px**2 * py**2
                    - 2 * px**2 * pz**2
                    + 2 * px**2 * rp**2
                    + 2 * px**2 * rs**2
                    - py**4
                    - 2 * py**2 * pz**2
                    + 2 * py**2 * rp**2
                    + 2 * py**2 * rs**2
                    - pz**4
                    + 2 * pz**2 * rp**2
                    - 2 * pz**2 * rs**2
                    - rp**4
                    + 2 * rp**2 * rs**2
                    - rs**4
                )
            )
            / (px**2 + 2 * px * rs + py**2 + pz**2 - rp**2 + rs**2)
        ),
        2
        * atan(
            (
                2 * py * rs
                + sqrt(
                    -(px**4)
                    - 2 * px**2 * py**2
                    - 2 * px**2 * pz**2
                    + 2 * px**2 * rp**2
                    + 2 * px**2 * rs**2
                    - py**4
                    - 2 * py**2 * pz**2
                    + 2 * py**2 * rp**2
                    + 2 * py**2 * rs**2
                    - pz**4
                    + 2 * pz**2 * rp**2
                    - 2 * pz**2 * rs**2
                    - rp**4
                    + 2 * rp**2 * rs**2
                    - rs**4
                )
            )
            / (px**2 + 2 * px * rs + py**2 + pz**2 - rp**2 + rs**2)
        ),
    ]

t = 0
while True:
    viz.display(robot.state.q)

    # Up and down
    # T_world_target = T_world_head @ tf.translation_matrix((0, 0, np.sin(t*3)*0.01))

    # Rolling
    T_world_target = T_world_head @ tf.rotation_matrix(np.sin(t*3)*0.6, (1, 0, 0))

    joints = {}
    for motor in motors:
        # Computing the target position in the motor frame
        T_target_branch = tf.translation_matrix(motor["branch_position"])
        T_world_branch = T_world_target @ T_target_branch
        T_motor_branch = (
            np.linalg.inv(motor["T_world_motor"]) @ T_world_target @ T_target_branch
        )
        branch_position = T_motor_branch[:3, 3]

        solutions = ik_branch(*branch_position)
        joints[motor["name"]] = placo.wrap_angle(motor["offset"] + solutions[motor["solution"]])
    
    # for name, value in joints.items():
    #     robot.set_joint(name, value)
    # robot.update_kinematics()
    solver.fk([0.0] + list(joints.values()))

    for k in range(1, 7):
        robot_frame_viz(robot, f"{k}", scale=0.2)

    robot.update_kinematics()
    time.sleep(0.02)
    t += 0.02
