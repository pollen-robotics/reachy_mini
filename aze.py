import json
import time
from importlib.resources import files

import numpy as np
from FramesViewer import Viewer
from reachy_mini_rust_kinematics import ReachyMiniRustKinematics
from scipy.spatial.transform import Rotation as R

import reachy_mini

fv = Viewer()
fv.start()

SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

assets_root_path: str = str(files(reachy_mini).joinpath("assets/"))
data_path = assets_root_path + "/kinematics_data.json"
data = json.load(open(data_path, "rb"))

automatic_body_yaw = True

head_z_offset = data["head_z_offset"]

kin = ReachyMiniRustKinematics(data["motor_arm_length"], data["rod_length"])


motors = data["motors"]
for motor in motors:
    kin.add_branch(
        motor["branch_position"],
        np.linalg.inv(motor["T_motor_world"]),  # type: ignore[arg-type]
        1 if motor["solution"] else -1,
    )

sleep_head_pose = SLEEP_HEAD_POSE.copy()
sleep_head_pose[:3, 3][2] += head_z_offset
kin.reset_forward_kinematics(sleep_head_pose)  # type: ignore[arg-type]

pose = np.array(
    [
        [0.49565772, -0.82289198, 0.2781861, -0.014483],
        [0.72819684, 0.56807686, 0.38383037, -0.01582323],
        [-0.47387355, 0.01206656, 0.880789, 0.13291363],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
body_yaw = 0.992
# body_yaw = 0

pose_viz = pose.copy()
pose_viz[:3, 3] += [0.1, 0.1, 0.1]
fv.push_frame(pose_viz, "ze")
root = np.eye(4)
# rotate root aroudn z axis by body_yaw
root[:3, :3] = R.from_euler("z", body_yaw).as_matrix()
root_viz = root.copy()
root_viz[:3, 3] += [0.1, 0.1, 0.1]
fv.push_frame(root_viz, "root_z")




# pose = np.eye(4)
# pose[:3, 3][2] += head_z_offset
reachy_joints = kin.inverse_kinematics_safe(
    pose,
    body_yaw=body_yaw,
    max_relative_yaw=np.deg2rad(65),
    max_body_yaw=np.deg2rad(160),
)

# stewart_joints = kin.inverse_kinematics(pose, body_yaw)  # type: ignore[arg-type]
# print(stewart_joints)
print(reachy_joints)
while True:
    time.sleep(1)
