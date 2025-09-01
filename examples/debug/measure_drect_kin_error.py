"""Reachy Mini Analytical Kinematics GUI Example."""

import os
import time

# Show histograms
import matplotlib.pyplot as plt
import numpy as np
from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz

from reachy_mini.kinematics import PlacoKinematics, ReachyMiniAnalyticKinematics

urdf_path = os.path.abspath(
    "../../src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
)

solver = PlacoKinematics(urdf_path, 0.02)
robot = solver.robot
robot.update_kinematics()

viz = robot_viz(robot)


robot_limits = []
for jn in solver.joints_names:
    robot_limits.append(robot.get_joint_limits(jn))

asolver = ReachyMiniAnalyticKinematics(urdf_path)

# initial joint angles
joints = solver.ik(solver.head_starting_pose)
position_errors = []
angular_errors = []
distances_from_initial = []
angular_distances_from_initial = []
solver_ik_times = []
solver_fk_times = []

np.random.seed(1578)

i = 0
while i < 2000:
    # Read GUI values
    px, py, pz = [np.random.uniform(-0.01, 0.01) for _ in range(3)]
    roll, pitch = [np.random.uniform(-np.deg2rad(30), np.deg2rad(30)) for _ in range(2)]
    yaw = np.random.uniform(-2.8, 2.8)
    body_yaw = -yaw  # + np.random.uniform(-np.deg2rad(20), np.deg2rad(20))

    # Compute the target transformation matrix in the world frame
    T_head_target = tf.translation_matrix((px, py, pz)) @ tf.euler_matrix(
        roll, pitch, yaw
    )

    joints1 = asolver.ik(pose=T_head_target, body_yaw=body_yaw)

    # Check for NaN values or incorrect joint count
    if len(joints1) != 7 or np.any(np.isnan(joints1)):
        print(f"Invalid IK analytic solution at iteration {i}, redoing iteration.")
        continue

    # joints = asolver.ik(pose=T_head_target, body_yaw=body_yaw)
    for _ in range(12):
        # Measure time for solver IK
        start_solver_ik = time.time()
        joints = solver.ik(pose=T_head_target, body_yaw=body_yaw)
        end_solver_ik = time.time()
        solver_ik_times.append(end_solver_ik - start_solver_ik)

    if len(joints) != 7 or np.any(np.isnan(joints)):
        print(f"Invalid IK solution at iteration {i}, redoing iteration.")
        continue

    for _ in range(20):
        # Measure time for solver FK
        start_solver_fk = time.time()
        final_pose = solver.fk(joints)
        end_solver_fk = time.time()
        solver_fk_times.append((end_solver_fk - start_solver_fk))

    # Compute pose error
    pos_error = np.linalg.norm(final_pose[:3, 3] - T_head_target[:3, 3])

    # Compute angular error (rotation matrix difference)
    R_final = final_pose[:3, :3]
    R_target = T_head_target[:3, :3]
    rot_diff = R_final @ R_target.T
    angle_error = np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1.0, 1.0))

    joints = [robot.get_joint(jn) for jn in solver.joints_names]
    skip = False
    for j in range(7):
        # Check if joint limits are respected
        if not (robot_limits[j][0] < joints[j] < robot_limits[j][1]):
            skip = True
            break
    if skip:
        continue

    if pos_error > 0.02 or angle_error > np.deg2rad(10):
        viz.display(robot.state.q)
        robot_frame_viz(robot, "head")
        frame_viz(
            "target",
            solver.head_starting_pose
            @ tf.translation_matrix((px, py, pz))
            @ tf.euler_matrix(roll, pitch, yaw),
        )
        print(px, py, pz, roll, pitch, yaw)
        # input("Press Enter to continue...")

    position_errors.append(pos_error)
    angular_errors.append(np.degrees(angle_error))

    # Compute distance from initial pose
    initial_pose = np.eye(4)
    distance_from_initial = np.linalg.norm(T_head_target[:3, 3])
    # Compute angular distance from initial pose
    R_initial = initial_pose[:3, :3]
    R_target = T_head_target[:3, :3]
    rot_diff_initial = R_target @ R_initial.T
    angular_distance_from_initial = np.arccos(
        np.clip((np.trace(rot_diff_initial) - 1) / 2, -1.0, 1.0)
    )

    distances_from_initial.append(distance_from_initial)
    angular_distances_from_initial.append(np.degrees(angular_distance_from_initial))

    i += 1

    if i % 100 == 0:
        print(f"Iteration {i} out of 1000")

print(f"Total iterations: {i}")
print(
    f"IK time: {np.mean(solver_ik_times) * 1000:.2f} ms (max {np.max(solver_ik_times) * 1000:.2f} ms)"
)
print(
    f"FK time: {np.mean(solver_fk_times) * 1000:.2f} ms (max {np.max(solver_fk_times) * 1000:.2f} ms)"
)

# plot the position and angular errors distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(np.array(position_errors) * 1000, bins=50)
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
plt.scatter(
    np.array(distances_from_initial) * 1000,
    np.array(position_errors) * 1000,
    alpha=0.5,
    label="Position Error (mm)",
)
plt.scatter(
    np.array(distances_from_initial) * 1000,
    angular_errors,
    alpha=0.5,
    c="r",
    label="Angular Error (deg)",
)
plt.title("Error vs Distance from Initial Position")
plt.xlabel("Translation distance from Initial Position (mm)")
plt.ylabel("Error (mm/deg)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(
    angular_distances_from_initial,
    np.array(position_errors) * 1000,
    alpha=0.5,
    label="Position Error (mm)",
)
plt.scatter(
    angular_distances_from_initial,
    angular_errors,
    alpha=0.5,
    c="r",
    label="Angular Error (deg)",
)
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
plt.hist(np.array(solver_ik_times) * 1000, bins=50)
plt.title("IK Solver Times")
plt.xlabel("Time (ms)")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(np.array(solver_fk_times) * 1000, bins=50)
plt.title("FK Solver Times")
plt.xlabel("Time (ms)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
