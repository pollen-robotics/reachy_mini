from reachy_mini_motor_controller import ReachyMiniMotorController
from reachy_mini.placo_kinematics import PlacoKinematics
from reachy_mini.analytic_kinematics import ReachyMiniAnalyticKinematics
from placo_utils.visualization import robot_viz
import numpy as np
import time
import pinocchio as pin

def main():
    
    
    urdf_path = "/home/gospar/pollen_robotics/reachy_mini/src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
    solver = PlacoKinematics(urdf_path, 0.02)
    robot = solver.robot
    robot.update_kinematics()
    data = robot.model.createData()
    # viz = robot_viz(robot)
    
    asolver = ReachyMiniAnalyticKinematics(robot=robot)
    
    
    

    # Initialize the motor controller (adjust port if needed)
    controller = ReachyMiniMotorController(serialport='/dev/ttyACM0')


    joint_names = ['1', '2', '3', '4', '5', '6']  # Keep only the Stewart platform motors
    joinit_ids = []
    column_idx = 0
    for joint_id in range(1, robot.model.njoints):  # Skip universe joint (ID=0)
        joint_name = robot.model.names[joint_id]
        nv = robot.model.joints[joint_id].nv
        if joint_name in joint_names:
            joinit_ids.append(column_idx)
        column_idx += nv
            
    print(f"Selected joint IDs: {joinit_ids}")
        
    # Details found here in the Specifications table
    # https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#Specifications
    k_Nm_to_mA = 1.47 / 0.52 * 1000  # Conversion factor from Nm to mA for the Stewart platform motors
    efficiency = 1.0  # Efficiency of the motors
    
    t0 = time.time()
    controller.enable_torque()  # Enable torque for the Stewart platform motors
    try:
        while time.time() - t0 < 15.0:  # Wait for the motors to stabilize
            motor_pos = controller.read_all_positions()[3:]

            solver.fk([0.0] + list(motor_pos))
            T_world_head = robot.get_T_world_frame("head")
            
            # viz.display(robot.state.q)
            time.sleep(0.01)
            #jac = asolver.jacobian(T_world_head)
            
            #torque = np.linalg.pinv(jac.T) @ np.array([0.0, 0.0, 9.81*0.1, 0.0, 0.0, 0.0]) # head weight is 0.5 kg
            torque = -pin.computeGeneralizedGravity(robot.model, data, robot.state.q)[joinit_ids] # Nm
            
            current = torque * k_Nm_to_mA / efficiency#mA
            print(f"Current: {current}")
            controller.set_stewart_platform_goal_current(np.round(current, 0).astype(int).tolist())

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Disabling torque and exiting.")
        controller.disable_torque()
        return
    
    controller.disable_torque()  # Enable torque

if __name__ == '__main__':
    
    main()