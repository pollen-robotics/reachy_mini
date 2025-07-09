from reachy_mini_motor_controller import ReachyMiniMotorController
from reachy_mini.placo_kinematics import PlacoKinematics
from reachy_mini.analytic_kinematics import ReachyMiniAnalyticKinematics
from placo_utils.visualization import robot_viz
import numpy as np
import time


def main():

    urdf_path = "src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
    solver = PlacoKinematics(urdf_path, 0.02)
    robot = solver.robot
    robot.update_kinematics()
    data = robot.model.createData()
    viz = robot_viz(robot)
    
    asolver = ReachyMiniAnalyticKinematics(robot=robot)
    
    # Initialize the motor controller (adjust port if needed)
    controller = ReachyMiniMotorController(serialport='/dev/ttyACM0')
    
    
    # Details found here in the Specifications table
    # https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#Specifications
    k_Nm_to_mA = 1.47 / 0.52 * 1000  # Conversion factor from Nm to mA for the Stewart platform motors
    efficiency = 1.0  # Efficiency of the motors
    
    t0 = time.time()
    controller.enable_torque()  # Enable torque for the Stewart platform motors
    
    # these values are based on the URDF and the robot's design
    # Center of mass and weight of the head
    com_head = np.array([0.0, 0.0, 0.03])  # Center of mass of the head
    weight_head = 0.07 * 9.81 # Weight of the head in N
    
    
    try:
        while time.time() - t0 < 150.0:  # Wait for the motors to stabilize
            motor_pos = controller.read_all_positions()[3:]

            solver.fk([0.0] + list(motor_pos))
            T_world_head = robot.get_T_world_frame("head")
            
            # viz.display(robot.state.q)
            time.sleep(0.01)
            jac = asolver.jacobian(T_world_head)
            # Compute torque vector due to head weight using the rotation matrix
            g = np.array([0, 0, weight_head])  # gravity vector in world frame
            r = T_world_head[:3, :3] @ com_head  # COM in world frame
            torque_world = np.cross(r, g)
            # Express torque in head frame
            torque_head = T_world_head[:3, :3].T @ torque_world
            torque_roll, torque_pitch, torque_yaw = torque_head[0], torque_head[1], torque_head[2]

            #torque = np.linalg.pinv(jac.T) @ np.array([0.0, 0.0, 9.81*0.07, 0.0, 0.0, 0.0]) # head weight is 0.1 kg
            torque = np.linalg.pinv(jac.T) @ np.array([0.0, 0.0, weight_head, torque_roll, torque_pitch, torque_yaw]) # head weight is 0.1 kg

            current = torque * k_Nm_to_mA / efficiency#mA
            controller.set_stewart_platform_goal_current(np.round(current, 0).astype(int).tolist())
            viz.display(robot.state.q)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disabling torque and exiting.")
        controller.disable_torque()
        return
    
    controller.disable_torque()  # Enable torque

if __name__ == '__main__':
    
    main()