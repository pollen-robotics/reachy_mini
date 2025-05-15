import mujoco
import time
import os
from pathlib import Path
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from stewart_little_control.mujoco_utils import get_homogeneous_matrix_from_euler
from stewart_little_control import IKWrapper

def get_homogeneous_matrix_from_euler(
    position: tuple = (0, 0, 0),  # (x, y, z)
    euler_angles: tuple = (0, 0, 0),  # (roll, pitch, yaw) in radians
    degrees: bool = False,
):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix

# --- Brute-force search function ---
def find_zero_pose_brute_force(
    ik_solver: IKWrapper,
    pos_ranges: list[tuple[float, float]],  # List of (min, max) for x, y, z
    angle_ranges: list[tuple[float, float]], # List of (min, max) for roll, pitch, yaw (in radians)
    num_steps_per_dim: int = 5, # Number of steps for each dimension
):
    """
    Brute-force searches for a 6-DOF pose that results in joint angles close to zero.

    Args:
        ik_solver: An initialized IKWrapper instance.
        pos_ranges: A list of 3 tuples [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        angle_ranges: A list of 3 tuples [(roll_min, roll_max), (pitch_min, pitch_max), (yaw_min, yaw_max)]
                      Angles should be in radians.
        num_steps_per_dim: The number of discrete steps to check for each dimension.
                           Total iterations will be (num_steps_per_dim ** 6).

    Returns:
        A tuple: (best_pose_params, min_error, best_joint_angles)
        best_pose_params: (x, y, z, roll, pitch, yaw) that resulted in min_error
        min_error: The minimum sum of squared joint angles found.
        best_joint_angles: The joint angles corresponding to the best_pose_params.
    """
    if len(pos_ranges) != 3 or len(angle_ranges) != 3:
        raise ValueError("Position and angle ranges must each contain 3 (min, max) tuples.")

    px_vals = np.linspace(pos_ranges[0][0], pos_ranges[0][1], num_steps_per_dim)
    py_vals = np.linspace(pos_ranges[1][0], pos_ranges[1][1], num_steps_per_dim)
    pz_vals = np.linspace(pos_ranges[2][0], pos_ranges[2][1], num_steps_per_dim)
    roll_vals = np.linspace(angle_ranges[0][0], angle_ranges[0][1], num_steps_per_dim)
    pitch_vals = np.linspace(angle_ranges[1][0], angle_ranges[1][1], num_steps_per_dim)
    yaw_vals = np.linspace(angle_ranges[2][0], angle_ranges[2][1], num_steps_per_dim)

    min_error = float('inf')
    best_pose_params = None
    best_joint_angles = None

    total_iterations = num_steps_per_dim ** 6
    current_iteration = 0
    start_time = time.time()

    print(f"Starting brute-force search with {total_iterations} iterations.")

    for x in px_vals:
        for y in py_vals:
            for z in pz_vals:
                for roll in roll_vals:
                    for pitch in pitch_vals:
                        for yaw in yaw_vals:
                            current_iteration += 1
                            if current_iteration % (max(1, total_iterations // 100)) == 0: # Print progress roughly 100 times
                                elapsed_time = time.time() - start_time
                                progress = current_iteration / total_iterations
                                estimated_total_time = elapsed_time / progress if progress > 0 else float('inf')
                                remaining_time = estimated_total_time - elapsed_time
                                print(f"Progress: {current_iteration}/{total_iterations} ({progress*100:.2f}%) "
                                      f"Min Error: {min_error:.4e}. ETA: {remaining_time:.2f}s")

                            current_pos = (x, y, z)
                            current_euler = (roll, pitch, yaw) # These are in radians

                            pose_matrix = get_homogeneous_matrix_from_euler(
                                position=current_pos,
                                euler_angles=current_euler,
                                degrees=False # Euler angles are already in radians
                            )

                            try:
                                joint_angles_rad = ik_solver.ik(pose_matrix)
                                if joint_angles_rad is None or not isinstance(joint_angles_rad, np.ndarray) or joint_angles_rad.size != 6:
                                    # print(f"Warning: IK returned None or invalid for pose: p={current_pos}, e={current_euler}")
                                    continue # Skip if IK fails or returns unexpected
                                if np.any(np.isnan(joint_angles_rad)):
                                    # print(f"Warning: IK returned NaN for pose: p={current_pos}, e={current_euler}")
                                    continue # Skip if IK returns NaNs
                            except Exception as e:
                                # print(f"Warning: IK solver failed for pose: p={current_pos}, e={current_euler}. Error: {e}")
                                continue # Skip on IK error

                            # Error: sum of squared joint angles
                            error = np.sum(np.square(joint_angles_rad))

                            if error < min_error:
                                min_error = error
                                best_pose_params = (x, y, z, roll, pitch, yaw)
                                best_joint_angles = joint_angles_rad
                                print(f"New best found! Error: {min_error:.4e}, Pose: (x={x:.3f}, y={y:.3f}, z={z:.3f}, "
                                      f"r={np.degrees(roll):.2f}°, p={np.degrees(pitch):.2f}°, y={np.degrees(yaw):.2f}°), "
                                      f"Joints: {np.degrees(best_joint_angles)}")


    end_time = time.time()
    print(f"Search finished in {end_time - start_time:.2f} seconds.")
    if best_pose_params:
        print("\n--- Best Result ---")
        print(f"Best Pose Parameters (x,y,z, roll,pitch,yaw):")
        print(f"  Position (m):      ({best_pose_params[0]:.4f}, {best_pose_params[1]:.4f}, {best_pose_params[2]:.4f})")
        print(f"  Orientation (deg): ({np.degrees(best_pose_params[3]):.2f}, {np.degrees(best_pose_params[4]):.2f}, {np.degrees(best_pose_params[5]):.2f})")
        print(f"  Orientation (rad): ({best_pose_params[3]:.4f}, {best_pose_params[4]:.4f}, {best_pose_params[5]:.4f})")
        print(f"Minimum Sum of Squared Joint Errors: {min_error:.6e}")
        print(f"Corresponding Joint Angles (deg): {np.degrees(best_joint_angles)}")
        print(f"Corresponding Joint Angles (rad): {best_joint_angles}")
    else:
        print("No suitable pose found within the given ranges and steps, or IK failed consistently.")

    return best_pose_params, min_error, best_joint_angles

# --- Main execution part (example) ---
if __name__ == "__main__":
    # Initialize your IKWrapper
    # This is a placeholder, replace with your actual IKWrapper initialization
    print("Initializing IK Wrapper...")
    try:
        ik_wrapper = IKWrapper()
        print("IK Wrapper initialized.")
    except Exception as e:
        print(f"Failed to initialize IKWrapper: {e}")
        print("Please ensure IKWrapper can be initialized correctly.")
        exit()

    # --- Define Search Parameters ---
    # These ranges are guesses. You'll need to adjust them based on your robot's workspace
    # and where you expect the zero-pose to be.
    # For a Stewart platform, the zero pose is often near its central, level configuration.

    # Positional ranges (in meters)
    # If your robot is small, these ranges might be appropriate.
    # If it's larger, or the zero point is far from the origin, expand these.
    pos_search_ranges = [
        (-0.2, 0.2),  # x range (e.g., -10mm to +10mm)
        (-0.2, 0.2),  # y range
        (-1.5, 0.6),   # z range (height, often positive for Stewart platforms, adjust based on your model)
    ]

    # Angular ranges (in radians)
    # Small angles around zero orientation.
    angle_search_ranges_rad = [
        (np.deg2rad(0), np.deg2rad(0)),  # roll range
        (np.deg2rad(0), np.deg2rad(0)),  # pitch range
        (np.deg2rad(0), np.deg2rad(0)),  # yaw range
    ]

    # Number of steps per dimension.
    # 5 steps -> 5^6 = 15,625 iterations
    # 7 steps -> 7^6 = 117,649 iterations
    # 10 steps -> 10^6 = 1,000,000 iterations (can take a while)
    # Start with a small number like 5 or 7 to test.
    search_steps_per_dim = 7 # Example: this will be 7^6 = 117,649 iterations

    print(f"Position search ranges (m): {pos_search_ranges}")
    print(f"Angle search ranges (deg): [({np.degrees(angle_search_ranges_rad[0][0]):.1f}, {np.degrees(angle_search_ranges_rad[0][1]):.1f}), "
          f"({np.degrees(angle_search_ranges_rad[1][0]):.1f}, {np.degrees(angle_search_ranges_rad[1][1]):.1f}), "
          f"({np.degrees(angle_search_ranges_rad[2][0]):.1f}, {np.degrees(angle_search_ranges_rad[2][1]):.1f})]")
    print(f"Steps per dimension: {search_steps_per_dim}")


    # Run the search
    best_pose, min_err, best_joints = find_zero_pose_brute_force(
        ik_wrapper,
        pos_search_ranges,
        angle_search_ranges_rad,
        search_steps_per_dim
    )

    # Now you can use `best_pose` in your simulation if a good one was found.
    # For example, to set it as a target:
    if best_pose:
        target_pos = best_pose[:3]
        target_euler_rad = best_pose[3:]
        print("\nTo use this pose in your simulation (example):")
        print(f"target_position = {target_pos}")
        print(f"target_euler_angles_rad = {target_euler_rad}")
        # target_hm_matrix = get_homogeneous_matrix_from_euler(target_pos, target_euler_rad)
        # target_joint_angles = ik_wrapper.ik(target_hm_matrix) # Should be close to best_joints
        # data.ctrl[:] = target_joint_angles
        
        # To verify the IK result with the found pose:
        verify_pose_matrix = get_homogeneous_matrix_from_euler(
            position=target_pos,
            euler_angles=target_euler_rad,
            degrees=False
        )
        verified_joint_angles = ik_wrapper.ik(verify_pose_matrix)
        print(f"\nVerification of IK with found best pose:")
        print(f"  Input Pose (x,y,z, r,p,y_rad): {target_pos}, {target_euler_rad}")
        print(f"  Resulting Joint Angles (rad): {verified_joint_angles}")
        print(f"  Resulting Joint Angles (deg): {np.degrees(verified_joint_angles)}")
        print(f"  Sum of Squared Errors: {np.sum(np.square(verified_joint_angles)):.4e}")

