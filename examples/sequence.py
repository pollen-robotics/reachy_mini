from reachy_mini.io import Client
from reachy_mini.command import ReachyMiniCommand
import time
import numpy as np
from scipy.spatial.transform import Rotation as R


"""
Warning: Don't run this lol
"""
client = Client()

while True:
    pose = np.eye(4)
    pose[:3, 3][2] = 0.177  # Set the height of the head

    t = 0
    t0 = time.time()
    s = time.time()
    while time.time() - s < 2.0:
        t = time.time() - t0
        euler_rot = np.array([0, 0.0, 0.7*np.sin(2*np.pi*0.5*t)])
        rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        pose[:3, :3] = rot_mat

        client.send_command(
            ReachyMiniCommand(
                head_pose=pose,
                antennas_orientation=[0, 0],
                offset_zero=False,
            )
        )
        time.sleep(0.01)

    s = time.time()
    while time.time() - s < 2.0:
        t = time.time() - t0
        euler_rot = np.array([0, 0.3*np.sin(2*np.pi*0.5*t), 0])
        rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        pose[:3, :3] = rot_mat
        client.send_command(
            ReachyMiniCommand(
                head_pose=pose,
                antennas_orientation=[0, 0],
                offset_zero=False,
            )
        )
        time.sleep(0.01)


    s = time.time()
    while time.time() - s < 2.0:
        t = time.time() - t0
        euler_rot = np.array([0.3*np.sin(2*np.pi*0.5*t), 0, 0])
        rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        pose[:3, :3] = rot_mat
        client.send_command(
            ReachyMiniCommand(
                head_pose=pose,
                antennas_orientation=[0, 0],
                offset_zero=False,
            )
        )
        time.sleep(0.01)

    s = time.time()
    while time.time() - s < 2.0:
        t = time.time() - t0
        pose = np.eye(4)
        pose[:3, 3][2] = 0.177
        pose[:3, 3][2] += 0.025*np.sin(2*np.pi*0.5*t)
        # euler_rot = np.array([0.3*np.sin(2*np.pi*0.5*t), 0, 0])
        # rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        # pose[:3, :3] = rot_mat
        client.send_command(
            ReachyMiniCommand(
                head_pose=pose,
                antennas_orientation=[0, 0],
                offset_zero=False,
            )
        )
        time.sleep(0.01)

    s = time.time()
    while time.time() - s < 2.0:
        t = time.time() - t0
        antennas = [0.5*np.sin(2*np.pi*0.5*t), -0.5*np.sin(2*np.pi*0.5*t)]
        client.send_command(
            ReachyMiniCommand(
                head_pose=pose,
                antennas_orientation=antennas,
                offset_zero=False,
            )
        )
        time.sleep(0.01)

    s = time.time()
    while time.time() - s < 5.0:
        t = time.time() - t0
        # euler_rot = np.array([0.3*np.sin(2*np.pi*0.5*t), 0, 0])
        # rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
        # pose[:3, :3] = rot_mat
        pose[:3, 3] = [
            0.015 * np.sin(2 * np.pi * 1.0 * t),
            0.015 * np.sin(2 * np.pi * 1.0 * t + np.pi / 2),
            0.177,
        ]
        client.send_command(
            ReachyMiniCommand(
                head_pose=pose,
                antennas_orientation=[0, 0],
                offset_zero=False,
            )
        )
        time.sleep(0.01)

    pose[:3, 3] = [0, 0, 0.177]
    client.send_command(
        ReachyMiniCommand(
            head_pose=pose,
            antennas_orientation=[0, 0],
            offset_zero=False,
        )
    )
    time.sleep(0.5)
    pose[:3, 3] = [0.02, 0.02, 0.177]
    client.send_command(
        ReachyMiniCommand(
            head_pose=pose,
            antennas_orientation=[0, 0],
            offset_zero=False,
        )
    )
    time.sleep(0.5)
    pose[:3, 3] = [0.00, 0.02, 0.177]
    euler_rot = np.array([0, 0, 0.5])
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat
    client.send_command(
        ReachyMiniCommand(
            head_pose=pose,
            antennas_orientation=[0, 0],
            offset_zero=False,
        )
    )
    time.sleep(0.5)
    pose[:3, 3] = [0.00, -0.02, 0.177]
    euler_rot = np.array([0, 0, -0.5])
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat
    client.send_command(
        ReachyMiniCommand(
            head_pose=pose,
            antennas_orientation=[0, 0],
            offset_zero=False,
        )
    )
    time.sleep(0.5)
    pose[:3, 3] = [0, 0, 0.177]
    client.send_command(
        ReachyMiniCommand(
            head_pose=pose,
            antennas_orientation=[0, 0],
            offset_zero=False,
        )
    )

    time.sleep(2)

# pose[:3, 3][0] = 0.02
# pose[:3, :3] = rot_mat
# exit()
