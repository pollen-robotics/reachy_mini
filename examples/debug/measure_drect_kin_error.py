"""Reachy Mini Analytical Kinematics GUI Example."""

import os
import time
import numpy as np

from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz

from reachy_mini.placo_kinematics import PlacoKinematics

urdf_path = os.path.abspath("src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf")
solver = PlacoKinematics(urdf_path, 0.02)
robot = solver.robot
robot.update_kinematics()

from reachy_mini.analytic_kinematics import ReachyMiniAnalyticKinematics

# viz = robot_viz(robot)

asolver = ReachyMiniAnalyticKinematics(urdf_path)

# initial joint angles
joints = solver.ik(solver.head_starting_pose)
position_errors = []
angular_errors = []
distances_from_initial = []
angular_distances_from_initial = []
solver_ik_times = []
solver_fk_times = []

i = 0
while i < 1000:
    # Read GUI values
    px, py, pz = [np.random.uniform(-0.01, 0.01) for _ in range(3)]
    roll, pitch = [np.random.uniform(-np.deg2rad(30), np.deg2rad(30)) for _ in range(2)]
    yaw = np.random.uniform(-3.1, 3.1)
    body_yaw = -yaw #+ np.random.uniform(-np.deg2rad(20), np.deg2rad(20))

    # Compute the target transformation matrix in the world frame
    T_head_target = tf.translation_matrix((px, py, pz)) @ tf.euler_matrix(
        roll, pitch, yaw
    )

    joints1 = asolver.ik(
        pose=T_head_target,
        body_yaw=body_yaw
    )

    # Check for NaN values or incorrect joint count
    if len(joints1) != 7 or np.any(np.isnan(joints1)):
        print(f"Invalid IK solution at iteration {i}, redoing iteration.")
        continue
    
    # Measure time for solver IK
    start_solver_ik = time.time()
    joints = solver.ik(pose=T_head_target, body_yaw=body_yaw)
    end_solver_ik = time.time()
    solver_ik_time = end_solver_ik - start_solver_ik

    if len(joints) != 7 or np.any(np.isnan(joints)):
        print(f"Invalid IK solution at iteration {i}, redoing iteration.")
        continue
    
    # Measure time for solver FK
    start_solver_fk = time.time()
    final_pose = solver.fk(joints)
    end_solver_fk = time.time()
    solver_fk_time = end_solver_fk - start_solver_fk

    # Save times for later analysis
    solver_ik_times.append(solver_ik_time)
    solver_fk_times.append(solver_fk_time)

    # Compute pose error
    pos_error = np.linalg.norm(final_pose[:3, 3] - T_head_target[:3, 3])

    # Compute angular error (rotation matrix difference)
    R_final = final_pose[:3, :3]
    R_target = T_head_target[:3, :3]
    rot_diff = R_final @ R_target.T
    angle_error = np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1.0, 1.0))

    if pos_error > 0.05 or angle_error > np.deg2rad(10):
        # avoid outliers
        continue
    
    position_errors.append(pos_error)
    angular_errors.append(np.degrees(angle_error))
    
    # Compute distance from initial pose
    initial_pose = np.eye(4)
    distance_from_initial = np.linalg.norm(T_head_target[:3, 3])
    # Compute angular distance from initial pose
    R_initial = initial_pose[:3, :3]
    R_target = T_head_target[:3, :3]
    rot_diff_initial = R_target @ R_initial.T
    angular_distance_from_initial = np.arccos(np.clip((np.trace(rot_diff_initial) - 1) / 2, -1.0, 1.0))

    distances_from_initial.append(distance_from_initial)
    angular_distances_from_initial.append(np.degrees(angular_distance_from_initial))
    
    i += 1
    
    if i % 100 == 0:
        print(f"Iteration {i} out of 1000")
        
# Show histograms
import matplotlib.pyplot as plt

# plot the position and angular errors distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(np.array(position_errors)*1000, bins=50)
plt.title("Position Error (mm)")
plt.xlabel("Error (mm)")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(angular_errors, bins=50)
plt.title("Angular Error (defg)")
plt.xlabel("Error (deg)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# plot the errors against the distance from the initial position
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(np.array(distances_from_initial)*1000, np.array(position_errors)*1000, alpha=0.5, label='Position Error (mm)')
plt.scatter(np.array(distances_from_initial)*1000, angular_errors, alpha=0.5, c='r', label='Angular Error (deg)')
plt.title("Error vs Distance from Initial Position")
plt.xlabel("Translation distance from Initial Position (mm)")
plt.ylabel("Error (mm/deg)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(angular_distances_from_initial, np.array(position_errors)*1000, alpha=0.5, label='Position Error (mm)')
plt.scatter(angular_distances_from_initial, angular_errors, alpha=0.5, c='r', label='Angular Error (deg)')
plt.title("Error vs Angular Distance from Initial Position")
plt.xlabel("Angular Distance from Initial Position (deg)")
plt.ylabel("Error (mm/deg)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# plot the fk/ik solver times
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(np.array(solver_ik_times)*1000, bins=50)
plt.title("IK Solver Times")
plt.xlabel("Time (ms)")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(np.array(solver_fk_times)*1000, bins=50)
plt.title("FK Solver Times")
plt.xlabel("Time (ms)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
