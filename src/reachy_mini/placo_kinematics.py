import numpy as np
import placo
import pinocchio as pin


class PlacoKinematics:
    def __init__(self, urdf_path: str, dt: float = 0.02):
        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)

        self.ik_solver = placo.KinematicsSolver(self.robot)
        self.ik_solver.mask_fbase(True)

        self.fk_solver = placo.KinematicsSolver(self.robot)
        self.fk_solver.mask_fbase(True)

        # they should be hard but we use soft to avoir singularities and 
        # tradeoff the precision for the robustness 
        constrant_type = "soft" # "hard" or "soft"

        # IK closing tasks
        ik_closing_task_1 = self.ik_solver.add_relative_position_task(
            "closing_1_1", "closing_1_2", np.zeros(3)
        )
        ik_closing_task_1.configure("closing_1",constrant_type, 1.0)

        ik_closing_task_2 = self.ik_solver.add_relative_position_task(
            "closing_2_1", "closing_2_2", np.zeros(3)
        )
        ik_closing_task_2.configure("closing_2",constrant_type, 1.0)

        ik_closing_task_3 = self.ik_solver.add_relative_position_task(
            "closing_3_1", "closing_3_2", np.zeros(3)
        )
        ik_closing_task_3.configure("closing_3",constrant_type, 1.0)

        ik_closing_task_4 = self.ik_solver.add_relative_position_task(
            "closing_4_1", "closing_4_2", np.zeros(3)
        )
        ik_closing_task_4.configure("closing_4",constrant_type, 1.0)

        ik_closing_task_5 = self.ik_solver.add_relative_position_task(
            "closing_5_1", "closing_5_2", np.zeros(3)
        )
        ik_closing_task_5.configure("closing_5",constrant_type, 1.0)

        # FK closing tasks
        fk_closing_task_1 = self.fk_solver.add_relative_position_task(
            "closing_1_1", "closing_1_2", np.zeros(3)
        )
        fk_closing_task_1.configure("closing_1",constrant_type, 1.0)

        fk_closing_task_2 = self.fk_solver.add_relative_position_task(
            "closing_2_1", "closing_2_2", np.zeros(3)
        )
        fk_closing_task_2.configure("closing_2",constrant_type, 1.0)

        fk_closing_task_3 = self.fk_solver.add_relative_position_task(
            "closing_3_1", "closing_3_2", np.zeros(3)
        )
        fk_closing_task_3.configure("closing_3",constrant_type, 1.0)

        fk_closing_task_4 = self.fk_solver.add_relative_position_task(
            "closing_4_1", "closing_4_2", np.zeros(3)
        )
        fk_closing_task_4.configure("closing_4",constrant_type, 1.0)

        fk_closing_task_5 = self.fk_solver.add_relative_position_task(
            "closing_5_1", "closing_5_2", np.zeros(3)
        )
        fk_closing_task_5.configure("closing_5",constrant_type, 1.0)

        self.joints_names = [
            "all_yaw",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]

        self.head_z_offset = 0.177  # offset for the head height

        # IK head task
        self.head_starting_pose = np.eye(4)
        self.head_starting_pose[:3, 3][2] = self.head_z_offset
        self.head_frame = self.ik_solver.add_frame_task("head", self.head_starting_pose)
        self.head_frame.configure("head", "soft", 1.0, 1.0)

        self.head_frame.T_world_frame = self.head_starting_pose

        # regularization
        self.ik_yaw_joint_task = self.ik_solver.add_joints_task()
        self.ik_yaw_joint_task.set_joints({"all_yaw": 0})
        self.ik_yaw_joint_task.configure("joints", "soft", 5e-5)

        self.ik_joint_task = self.ik_solver.add_joints_task()
        self.ik_joint_task.set_joints({f"{k}": 0 for k in range(1, 7)})
        self.ik_joint_task.configure("joints", "soft", 1e-5)

        self.ik_solver.enable_velocity_limits(True)
        self.ik_solver.dt = dt

        # FK joint task
        self.head_joints_task = self.fk_solver.add_joints_task()
        self.head_joints_task.configure("joints", "soft", 1.0)

        # regularization
        self.fk_yaw_joint_task = self.fk_solver.add_joints_task()
        self.fk_yaw_joint_task.set_joints({"all_yaw": 0})
        self.fk_yaw_joint_task.configure("joints", "soft", 5e-5)

        self.fk_joint_task = self.fk_solver.add_joints_task()
        self.fk_joint_task.set_joints({f"{k}": 0 for k in range(1, 7)})
        self.fk_joint_task.configure("joints", "soft", 1e-5)

        # self.fk_solver.enable_velocity_limits(True)
        self.fk_solver.dt = dt
        
        # setup the collision model
        self.config_collision_model()
        
    """
    Inverse Kinematics (IK) for the head of the robot.
    
    :param pose: 4x4 pose matrix representing the desired position and orientation of the head.
    :param check_collision: If True, checks for collisions after solving IK. (default: False)
    
    :return: A list of joints corresponding to the head's position and orientation, or None if a collision is detected.
    """
    
    def ik(self, pose, check_collision: bool = False):
        _pose = pose.copy()
        _pose[:3, 3][2] += self.head_z_offset  # offset the height of the head
        self.head_frame.T_world_frame = _pose
        
        
        # diminue the softness of the all_yaw joint if the head is below the torso
        # 0 at -1cm and 5e-5 at 0cm
        z_coord = _pose[:3, 3][2] - 0.177
        weight = 5e-5* (0.01 + z_coord)/0.01 if z_coord >= -0.01 else 0.0
        self.ik_yaw_joint_task.configure("joints", "soft", weight)
        
        q = self.robot.state.q.copy()
        for _ in range(10):
            try:
                self.ik_solver.solve(True)  # False to not update the kinematics            
            except Exception as e:
                print(f"IK solver failed: {e}")
                self.robot.state.q = q
            self.robot.update_kinematics()
            
        # verify that there is no collision
        if check_collision and self.compute_collision():
            print("Collision detected, stopping ik...")
            self.robot.state.q = q  # revert to the previous state
            self.robot.update_kinematics()
            return None

        joints = []
        for joint_name in self.joints_names:
            joint = self.robot.get_joint(joint_name)
            joints.append(joint)

        return joints

    """
    Forward Kinematics (FK) for the head of the robot.
    
    :param joints_angles: A list of joint angles corresponding to the head's position and orientation.
    :param check_collision: If True, checks for collisions after solving FK. (default: False)
    
    :return: A 4x4 transformation matrix representing the head's position and orientation in the world frame, or None if a collision is detected.
    """
    def fk(self, joints_angles, check_collision  = False):
        self.head_joints_task.set_joints(
            {
                "all_yaw": joints_angles[0],
                "1": joints_angles[1],
                "2": joints_angles[2],
                "3": joints_angles[3],
                "4": joints_angles[4],
                "5": joints_angles[5],
                "6": joints_angles[6],
            }
        )

        q = self.robot.state.q.copy()
        for _ in range(10):
            self.fk_solver.solve(True)
            self.robot.update_kinematics()
        

        if check_collision and self.compute_collision():
            print("Collision detected, stopping ik...")
            self.robot.state.q = q  # revert to the previous state
            return None
            

        T_world_head = self.robot.get_T_world_frame("head")
        T_world_head[:3, 3][2] -= self.head_z_offset  # offset the height of the head
        return T_world_head

    def config_collision_model(self):
        model = self.robot.model
        geom_model = self.robot.collision_model
        
        name_torso_collider = "dc15_a01_case_b_dummy_10"
        names_head_colliders = ["pp01063_stewart_plateform_7", "pp01063_stewart_plateform_11"]
        
        
        id_torso_collider = 12# geom_model.getGeometryObjectId(name_torso_collider)
        id_head_colliders = [74, 78]#[geom_model.getGeometryObjectId(name) for name in names_head_colliders]
        
        for i in id_head_colliders:
            geom_model.addCollisionPair(pin.CollisionPair(id_torso_collider, i))  # torso with head colliders
        

    def compute_collision(self):
        """
        Compute the collision between the robot and the environment.
        :return: True if there is a collision, False otherwise.
        """
        collision_data = self.robot.collision_model.createData()
        model = self.robot.model
        data = self.robot.model.createData()
        # pin.forwardKinematics(model, data, self.robot.state.q)
        # pin.updateFramePlacements(model, data)
        return pin.computeCollisions(self.robot.model, data, self.robot.collision_model, collision_data, self.robot.state.q)
        
    def manipulability(self):
        """
        Compute the manipulability of the robot.
        :return: The manipulability of the robot.
        """
        #jac = self.robot.frame_jacobian("head", "local_world_aligned")
        joint_ids = [self.robot.model.getJointId(joint_name)+4 for joint_name in self.joints_names]
        jac = self.robot.frame_jacobian("passive_7_link_y", "local_world_aligned")
        #print joint names
        # jid = []
        # for j in self.robot.model.frames:
        #     if self.robot.model.getJointId(j.name) <= self.robot.model.nq: # and j.name in self.joints_names:
        #         print(j.name, ":", self.robot.model.getJointId(j.name))
        #         #jid.append(self.robot.model.getJointId(j.name))

        # print all joint names and their IDs
        for j in self.robot.model.joints:
            if j.id <= self.robot.model.nq:
                print(self.robot.model.names[j.id], ":", j.id)
        
    
        # column_idx = 0
        # print("Joint-to-Jacobian Column Mapping:")
        # for joint_id in range(1, self.robot.model.njoints):  # Skip universe joint (ID=0)
        #     joint_name = self.robot.model.names[joint_id]
        #     nv = self.robot.model.joints[joint_id].nv
        #     print(f"Joint ID: {joint_id}, Name: {joint_name}, Jacobian columns: {column_idx} to {column_idx + nv - 1}")
        #     column_idx += nv
            
        # print("Joint IDs:", jid)
        #joint_ids = [self.robot.model.getJointId(joint_name) for joint_name in ["6"]]
        print("Joint IDs:", joint_ids)
        jac = jac  # Select only the joints of the head
        print("Jacobian matrix:\n", np.round(jac[:,joint_ids],2))
        jacobian_matrix1 = jac[:3, joint_ids].copy()  # Take only the position
        jacobian_matrix2 = jac[3:, joint_ids].copy()  # Take only the rotation
        return np.linalg.svd(jacobian_matrix1, compute_uv=False), np.linalg.svd(jacobian_matrix2, compute_uv=False)