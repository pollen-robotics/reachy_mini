import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMiniApp
from reachy_mini.reachy_mini import ReachyMini


class HelloApp(ReachyMiniApp):
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        t0 = time.time()
        while not stop_event.is_set():
            t = time.time() - t0

            target = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)

            yaw = target
            head = np.eye(4)
            head[:3, :3] = R.from_euler("xyz", [0, 0, yaw], degrees=False).as_matrix()

            reachy_mini.set_position(head=head, antennas=np.array([target, -target]))

            time.sleep(0.01)
