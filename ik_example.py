from stewart_little_control import IKWrapper
import numpy as np

pose = np.eye(4)

ik_wrapper = IKWrapper()
angles_deg = ik_wrapper.ik(pose, degrees=True)
angles_rad = ik_wrapper.ik(pose)
print("Angles in degrees:", angles_deg)
print("Angles in radians:", angles_rad)
