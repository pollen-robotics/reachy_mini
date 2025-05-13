from stewart_little_control.stewart_controller_noplot import Stewart_Platform
import numpy as np
from scipy.spatial.transform import Rotation as R

class IKWrapper:
    def __init__(self):

        r_B = 95.226  # radius of base
        r_P = 46.188  # radius of platform
        lhl = 38  # servo horn length
        # ldl = 180 + 2 * 4.2  # length of the rod
        ldl = 104 + 2 * 4.2  # length of the rod
        psi_B = np.deg2rad(21.56)  # Half of angle between two anchors on the base
        psi_P = np.deg2rad(30)  # Half of angle between two anchors on the platform

        self.platform = Stewart_Platform(r_B, r_P, lhl, ldl, psi_B, psi_P, 0)

        self.x_range = [-0.015, 0.015]
        self.y_range = [-0.015, 0.015]
        self.z_range = [-0.02, 0.02]
        self.roll_range = [-np.deg2rad(15), np.deg2rad(15)]
        self.pitch_range = [-np.deg2rad(15), np.deg2rad(15)]
        self.yaw_range = [-np.deg2rad(15), np.deg2rad(15)]

    def clip_pose(self, pose):
        pos = pose[:3, 3]
        euler = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)

        pos[0] = np.clip(pos[0], self.x_range[0], self.x_range[1])
        pos[1] = np.clip(pos[1], self.y_range[0], self.y_range[1])
        pos[2] = np.clip(pos[2], self.z_range[0], self.z_range[1])

        euler[0] = np.clip(euler[0], self.roll_range[0], self.roll_range[1])
        euler[1] = np.clip(euler[1], self.pitch_range[0], self.pitch_range[1])
        euler[2] = np.clip(euler[2], self.yaw_range[0], self.yaw_range[1])
        rot_mat = R.from_euler("xyz", euler, degrees=False).as_matrix()
        pose[:3, 3] = pos
        pose[:3, :3] = rot_mat
        return pose

    def ik(self, pose, degrees=False):
        pose = self.clip_pose(pose)
        pos = pose[:3, 3]
        pos_mm = pos * 1000

        rot_mat = pose[:3, :3]
        rot_euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=False)

        angles_rad = self.platform.calculate(np.array(pos_mm), np.array(rot_euler))

        if degrees:
            return np.rad2deg(angles_rad)
        else:
            return angles_rad
