import time  # noqa: D100

import numpy as np

from reachy_mini.kinematics import CPPAnalyticKinematics

cpp_kin = CPPAnalyticKinematics(
    urdf_path="../../src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
)

ik_times = []
fk_times = []

pose = np.eye(4)
t0 = time.time()
for i in range(1000):
    t = time.time() - t0
    pose[:3, 3][2] += 0.1 * np.sin(2 * np.pi * 0.5 * t)

    ik_t0 = time.time()
    joints = cpp_kin.ik(pose)
    ik_time = time.time() - ik_t0
    ik_times.append(ik_time)

    fk_t0 = time.time()
    cpp_kin.fk(np.double(joints), no_iterations=1)
    fk_time = time.time() - fk_t0
    fk_times.append(fk_time)


print(f"mean IK time (µs): {np.mean(ik_times) * 1e6:.1f} ± {np.std(ik_times) * 1e6:.1f}")
print(f"mean FK time (µs): {np.mean(fk_times) * 1e6:.1f} ± {np.std(fk_times) * 1e6:.1f}")