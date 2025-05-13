from stewart_little_control.stewart_little_control.io_330 import Dxl330IO
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from stewart_little_control.stewart_little_control.ik_wrapper import IKWrapper


class StewartLittleControl:
    def __init__(self):
        self.ik_wrapper = IKWrapper()

        self.ids = [1, 2, 3, 4, 5, 6]
        self.sign = [-1, 1, -1, 1, -1, 1]
        self.offset_deg = -22.5

        self.dxl_io = Dxl330IO("/dev/ttyACM0", baudrate=1000000, use_sync_read=True)

        self.dxl_io.enable_torque(self.ids)

        zero = {}
        for id in self.ids:
            self.dxl_io.set_pid_gain({id: [400, 0, 0]})
        for id in self.ids:
            zero[id] = 0
        self.dxl_io.set_goal_position(zero)
        time.sleep(2)

        for id in self.ids:
            self.dxl_io.set_pid_gain({id: [2000, 0, 0]})

    def set_pose(self, pose):
        angles_deg = self.ik_wrapper.ik(pose, degrees=True)
        target = {}
        for i, id in enumerate(self.ids):
            goal_pos = (angles_deg[i] + self.offset_deg) * self.sign[i]
            target[id] = goal_pos

        self.dxl_io.set_goal_position(target)


if __name__ == "__main__":
    control = StewartLittleControl()

    s = time.time()
    while True:
        t = time.time() - s
        pose = np.eye(4)
        euler_rot = [
            0,
            0,
            # 0.2 * np.sin(2 * np.pi * 0.5 * t),
            0.2 * np.sin(2 * np.pi * 0.5 * t + np.pi),
        ]
        rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        pose[:3, :3] = rot_mat

        control.set_pose(pose)
        time.sleep(0.01)
