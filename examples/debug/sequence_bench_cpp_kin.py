"""Reachy Mini Motion Sequence Example."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini.kinematics import CPPAnalyticKinematics

cpp_kin = CPPAnalyticKinematics(
    urdf_path="../../src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
)

ik_times = []
fk_times = []

pose = np.eye(4)

t = 0
t0 = time.time()
s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    euler_rot = np.array([0, 0.0, 0.7 * np.sin(2 * np.pi * 0.5 * t)])
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat

    ik_t0 = time.time()
    joints = cpp_kin.ik(pose)
    ik_time = time.time() - ik_t0
    ik_times.append(ik_time)

    fk_t0 = time.time()
    cpp_kin.fk(np.double(joints), no_iterations=1)
    fk_time = time.time() - fk_t0
    fk_times.append(fk_time)

    time.sleep(0.01)

s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    euler_rot = np.array([0, 0.3 * np.sin(2 * np.pi * 0.5 * t), 0])
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat

    ik_t0 = time.time()
    joints = cpp_kin.ik(pose)
    ik_time = time.time() - ik_t0
    ik_times.append(ik_time)

    fk_t0 = time.time()
    cpp_kin.fk(np.double(joints), no_iterations=1)
    fk_time = time.time() - fk_t0
    fk_times.append(fk_time)

    time.sleep(0.01)

s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    euler_rot = np.array([0.3 * np.sin(2 * np.pi * 0.5 * t), 0, 0])
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat

    ik_t0 = time.time()
    joints = cpp_kin.ik(pose)
    ik_time = time.time() - ik_t0
    ik_times.append(ik_time)

    fk_t0 = time.time()
    cpp_kin.fk(np.double(joints), no_iterations=1)
    fk_time = time.time() - fk_t0
    fk_times.append(fk_time)

    time.sleep(0.01)

s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    pose = np.eye(4)
    pose[:3, 3][2] += 0.025 * np.sin(2 * np.pi * 0.5 * t)

    ik_t0 = time.time()
    joints = cpp_kin.ik(pose)
    ik_time = time.time() - ik_t0
    ik_times.append(ik_time)

    fk_t0 = time.time()
    cpp_kin.fk(np.double(joints), no_iterations=1)
    fk_time = time.time() - fk_t0
    fk_times.append(fk_time)

    time.sleep(0.01)

s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    antennas = [
        0.5 * np.sin(2 * np.pi * 0.5 * t),
        -0.5 * np.sin(2 * np.pi * 0.5 * t),
    ]

    ik_t0 = time.time()
    joints = cpp_kin.ik(pose)
    ik_time = time.time() - ik_t0
    ik_times.append(ik_time)

    fk_t0 = time.time()
    cpp_kin.fk(np.double(joints), no_iterations=1)
    fk_time = time.time() - fk_t0
    fk_times.append(fk_time)

    time.sleep(0.01)

s = time.time()
while time.time() - s < 5.0:
    t = time.time() - t0
    pose[:3, 3] = [
        0.015 * np.sin(2 * np.pi * 1.0 * t),
        0.015 * np.sin(2 * np.pi * 1.0 * t + np.pi / 2),
        0.0,
    ]

    ik_t0 = time.time()
    joints = cpp_kin.ik(pose)
    ik_time = time.time() - ik_t0
    ik_times.append(ik_time)

    fk_t0 = time.time()
    cpp_kin.fk(np.double(joints), no_iterations=1)
    fk_time = time.time() - fk_t0
    fk_times.append(fk_time)

    time.sleep(0.01)

pose[:3, 3] = [0, 0, 0.0]

ik_t0 = time.time()
joints = cpp_kin.ik(pose)
ik_time = time.time() - ik_t0
ik_times.append(ik_time)

fk_t0 = time.time()
cpp_kin.fk(np.double(joints), no_iterations=1)
fk_time = time.time() - fk_t0
fk_times.append(fk_time)


time.sleep(0.5)

pose[:3, 3] = [0.02, 0.02, 0.0]

ik_t0 = time.time()
joints = cpp_kin.ik(pose)
ik_time = time.time() - ik_t0
ik_times.append(ik_time)

fk_t0 = time.time()
cpp_kin.fk(np.double(joints), no_iterations=1)
fk_time = time.time() - fk_t0
fk_times.append(fk_time)

time.sleep(0.5)

pose[:3, 3] = [0.00, 0.02, 0.0]
euler_rot = np.array([0, 0, 0.5])
rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
pose[:3, :3] = rot_mat

ik_t0 = time.time()
joints = cpp_kin.ik(pose)
ik_time = time.time() - ik_t0
ik_times.append(ik_time)

fk_t0 = time.time()
cpp_kin.fk(np.double(joints), no_iterations=1)
fk_time = time.time() - fk_t0
fk_times.append(fk_time)

time.sleep(0.5)

pose[:3, 3] = [0.00, -0.02, 0.0]
euler_rot = np.array([0, 0, -0.5])
rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
pose[:3, :3] = rot_mat

ik_t0 = time.time()
joints = cpp_kin.ik(pose)
ik_time = time.time() - ik_t0
ik_times.append(ik_time)

fk_t0 = time.time()
cpp_kin.fk(np.double(joints), no_iterations=1)
fk_time = time.time() - fk_t0
fk_times.append(fk_time)


pose[:3, 3] = [0, 0, 0.0]

ik_t0 = time.time()
joints = cpp_kin.ik(pose)
ik_time = time.time() - ik_t0
ik_times.append(ik_time)

fk_t0 = time.time()
cpp_kin.fk(np.double(joints), no_iterations=1)
fk_time = time.time() - fk_t0
fk_times.append(fk_time)


print(
    f"mean IK time (µs): {np.mean(ik_times) * 1e6:.1f} ± {np.std(ik_times) * 1e6:.1f}"
)
print(
    f"mean FK time (µs): {np.mean(fk_times) * 1e6:.1f} ± {np.std(fk_times) * 1e6:.1f}"
)
