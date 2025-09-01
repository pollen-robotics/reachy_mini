import os  # noqa: D100

import numpy as np
from placo_utils.tf import tf

from reachy_mini.kinematics import NNKinematics, PlacoKinematics

urdf_path = os.path.abspath(
    "../../src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
)

placo_kinematics = PlacoKinematics(urdf_path)
placo_kinematics.robot.update_kinematics()
nn_kinematics = NNKinematics("../../src/reachy_mini/assets/models/")

i = -1
while i < 2000:
    i += 1
    px, py, pz = [np.random.uniform(-0.01, 0.01) for _ in range(3)]
    roll, pitch = [np.random.uniform(-np.deg2rad(30), np.deg2rad(30)) for _ in range(2)]
    yaw = np.random.uniform(-2.8, 2.8)
    # yaw = 0
    body_yaw = -yaw  # + np.random.uniform(-np.deg2rad(20), np.deg2rad(20))
    body_yaw = 0

    T_head_target = tf.translation_matrix((px, py, pz)) @ tf.euler_matrix(
        roll, pitch, yaw
    )

    placo_result = placo_kinematics.ik(
        pose=T_head_target, body_yaw=body_yaw, no_iterations=20
    )
    nn_result = nn_kinematics.ik(pose=T_head_target, body_yaw=body_yaw)

    print(f"Placo Kinematics Result: {np.around(placo_result, 3)}")
    print(f"NN Kinematics Result: {np.around(nn_result, 3)}")
    print("==")
