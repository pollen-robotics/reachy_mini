"""Reachy Mini Gravity Compensation Direct Control Example."""

import time

import numpy as np
from placo_utils.visualization import robot_viz
from reachy_mini_motor_controller import ReachyMiniMotorController

from reachy_mini.kinematics import PlacoKinematics


def main():
    """Run a demo to compensate the gravity of the Reachy Mini platform."""
    urdf_path = "src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
    solver = PlacoKinematics(urdf_path, 0.02)
    robot = solver.robot
    robot.update_kinematics()
    viz = robot_viz(robot)

    # Initialize the motor controller (adjust port if needed)
    controller = ReachyMiniMotorController(serialport="/dev/ttyACM0")

    # Details found here in the Specifications table
    # https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#Specifications
    # the torque constant seems to be nonlinear and is not constant!!!!
    k_Nm_to_mA = (
        1.47 / 0.52 * 1000
    )  # Conversion factor from Nm to mA for the Stewart platform motors
    efficiency = 1.0  # Efficiency of the motors
    # torque constant correction factor
    correction_factor = 3.0  # This number is valid for currents under 30mA

    t0 = time.time()
    controller.disable_torque()  # Disable torque for the Stewart platform motors
    controller.set_stewart_platform_operating_mode(
        0
    )  # Set operation mode to torque control
    controller.enable_torque()  # Enable torque for the Stewart platform motors

    print("Robot is now compliant. Press Ctrl+C to exit.")
    try:
        while time.time() - t0 < 150.0:  # Wait for the motors to stabilize
            motor_pos = controller.read_all_positions()
            head_pos = [motor_pos[0]] + motor_pos[
                3:
            ]  # Extract head motor positions (all_yaw, 1, 2, 3, 4, 5, 6)

            # compute the gravity torque
            gravity_torque = solver.compute_gravity_torque(head_pos)
            # the target motor current
            current = gravity_torque * k_Nm_to_mA / efficiency / correction_factor  # mA
            # set the current to the motors
            controller.set_stewart_platform_goal_current(
                np.round(current[1:], 0).astype(int).tolist()
            )
            viz.display(robot.state.q)

    except KeyboardInterrupt:
        pass

    print("Robot is stiff again.")
    controller.disable_torque()  # Enable torque
    controller.set_stewart_platform_operating_mode(
        3
    )  # Set operation mode to torque control
    controller.enable_torque()  # Enable torque for the Stewart platform motors


if __name__ == "__main__":
    main()
