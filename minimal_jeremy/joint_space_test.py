from io_330 import Dxl330IO
import time
import numpy as np

io = Dxl330IO("COM6", baudrate=1000000, use_sync_read=True)
ids = [1, 2, 3, 4, 5, 6]
dxl_io.enable_torque(ids)



while True:
    target = {}
    angle = 10*np.sin(2*np.pi*f*time.time())
    for i, id in enumerate(ids):
        goal_pos = angle
        target[id] = goal_pos

    dxl_io.set_goal_position(target)
    time.sleep(0.01)
