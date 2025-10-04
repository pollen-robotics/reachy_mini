"""Rerun logging for Reachy Mini.

This module provides functionality to log the state of the Reachy Mini robot to Rerun,
 a tool for visualizing and debugging robotic systems.

It includes methods to log joint positions, camera images, and other relevant data.
"""

import json
import logging
import os
import tempfile
import time
from threading import Event, Thread

import cv2
import numpy as np
import requests
import rerun as rr
from rerun_loader_urdf import URDFLogger
from urdf_parser_py import urdf

from reachy_mini import ReachyMini
from reachy_mini.kinematics.placo_kinematics import PlacoKinematics


class Rerun:
    """Rerun logging for Reachy Mini."""

    def __init__(
        self,
        reachymini: ReachyMini,
        app_id: str = "reachy_mini_rerun",
        spawn: bool = True,
    ):
        """Initialize the Rerun logging for Reachy Mini.

        Args:
            backend (Backend): The backend to use for communication with the robot.
            app_id (str): The application ID for Rerun. Defaults to reachy_mini_daemon.
            spawn (bool): If True, spawn the Rerun server. Defaults to True.
            video (bool): If True, enable video capture from the camera. Defaults to False.

        """
        rr.init(app_id, spawn=spawn)
        self.app_id = app_id
        self._reachymini = reachymini
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(reachymini.logger.getEffectiveLevel())

        self.recording = rr.get_global_data_recording()

        script_dir = os.path.dirname(os.path.abspath(__file__))

        urdf_path = os.path.join(
            script_dir, "../descriptions/reachy_mini/urdf/robot.urdf"
        )
        asset_path = os.path.join(script_dir, "../descriptions/reachy_mini/urdf")

        fixed_urdf = self.set_absolute_path_to_urdf(urdf_path, asset_path)
        self.logger.debug(
            f"Using URDF file: {fixed_urdf} with absolute paths for Rerun."
        )

        self.head_kinematics = PlacoKinematics(fixed_urdf)

        self.urdf_logger = URDFLogger(fixed_urdf, "ReachyMini")
        self.urdf_logger.log(recording=self.recording)

        self.running = Event()
        self.thread_log_camera = Thread(target=self.log_camera, daemon=True)
        self.thread_log_movements = Thread(target=self.log_movements, daemon=True)
        # self.thread_log_rerun_info = Thread(target=self.log_rerun_info, daemon=True)
        # self.thread_log_mouvement = Thread(target=self.log_movement, daemon=True)

    def set_absolute_path_to_urdf(self, urdf_path: str, abs_path: str):
        """Set the absolute paths in the URDF file. Rerun cannot read the "package://" paths."""
        with open(urdf_path, "r") as f:
            urdf_content = f.read()
        urdf_content_mod = urdf_content.replace("package://", f"file://{abs_path}/")

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".urdf") as tmp_file:
            tmp_file.write(urdf_content_mod)
            return tmp_file.name

    def start(self):
        """Start the Rerun logging thread."""
        # self.thread_log_rerun_info.start()
        # self.thread_log_mouvement.start()
        self.thread_log_camera.start()
        self.thread_log_movements.start()

    def stop(self):
        """Stop the Rerun logging thread."""
        self.running.set()

    def _get_joint(self, joint_name: str) -> urdf.Joint:
        for j in self.urdf_logger.urdf.joints:
            if j.name == joint_name:
                return j
        raise RuntimeError("Invalid joint name")

    def _set_rod_rotation(
        self, joint_name: str, joint, joint_path, urdf_offset: list, id_rotation: int
    ):
        joint_rot = self.head_kinematics.get_joint(joint_name)
        urdf_offset[id_rotation] += joint_rot
        joint.origin.rotation = urdf_offset

        self.urdf_logger.log_joint(joint_path, joint=joint, recording=self.recording)

    def log_camera(self):
        """Log the camera image to Rerun."""
        if self._reachymini.media.camera is None:
            self.logger.warning("Camera is not initialized.")
            return

        self.logger.info("Starting camera logging to Rerun.")

        while not self.running.is_set():
            rr.set_time("reachymini", timestamp=time.time(), recording=self.recording)
            frame = self._reachymini.media.get_frame()
            if frame is not None:
                cam_name = self._get_joint("camera_optical_frame")
                cam_joint = self.urdf_logger.joint_entity_path(cam_name)

            K = np.array(
                [
                    [550.3564, 0.0, 638.0112],
                    [0.0, 549.1653, 364.589],
                    [0.0, 0.0, 1.0],
                ]
            )

            cam_name.origin.rotation = [3.14159, 1.0472, 3.14159]
            cam_name.origin.position = [0.0244171, -0.0524, 0.0147383]
            self.urdf_logger.log_joint(
                cam_joint, joint=cam_name, recording=self.recording
            )

            rr.log(
                f"{cam_joint}/image",
                rr.Pinhole(
                    image_from_camera=rr.datatypes.Mat3x3(K),
                    width=frame.shape[1],
                    height=frame.shape[0],
                    image_plane_distance=0.8,
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )

            # ToDo: this is suboptimal since the camera outputs a MJPEG stream
            # use alternative to opencv
            ret, encoded_image = cv2.imencode(".jpg", frame)
            if ret:
                rr.log(
                    f"{cam_joint}/image",
                    rr.EncodedImage(contents=encoded_image, media_type="image/jpeg"),
                )
            else:
                self.logger.error("Failed to encode frame to JPEG.")

            time.sleep(0.3)  # ~30fps

    def log_movements(self):
        """Log the movement data to Rerun."""
        antenna_left = self._get_joint("left_antenna")
        antenna_left_joint = self.urdf_logger.joint_entity_path(antenna_left)
        antenna_right = self._get_joint("right_antenna")
        antenna_right_joint = self.urdf_logger.joint_entity_path(antenna_right)

        motor_1 = self._get_joint("1")
        motor_1_joint = self.urdf_logger.joint_entity_path(motor_1)
        motor_2 = self._get_joint("2")
        motor_2_joint = self.urdf_logger.joint_entity_path(motor_2)
        motor_3 = self._get_joint("3")
        motor_3_joint = self.urdf_logger.joint_entity_path(motor_3)
        motor_4 = self._get_joint("4")
        motor_4_joint = self.urdf_logger.joint_entity_path(motor_4)
        motor_5 = self._get_joint("5")
        motor_5_joint = self.urdf_logger.joint_entity_path(motor_5)
        motor_6 = self._get_joint("6")
        motor_6_joint = self.urdf_logger.joint_entity_path(motor_6)
        motor_yaw = self._get_joint("all_yaw")
        motor_yaw_joint = self.urdf_logger.joint_entity_path(motor_yaw)

        passive_1_x = self._get_joint("passive_1_x")
        passive_1_x_joint = self.urdf_logger.joint_entity_path(passive_1_x)
        passive_1_y = self._get_joint("passive_1_y")
        passive_1_y_joint = self.urdf_logger.joint_entity_path(passive_1_y)
        passive_1_z = self._get_joint("passive_1_z")
        passive_1_z_joint = self.urdf_logger.joint_entity_path(passive_1_z)

        passive_2_x = self._get_joint("passive_2_x")
        passive_2_x_joint = self.urdf_logger.joint_entity_path(passive_2_x)
        passive_2_y = self._get_joint("passive_2_y")
        passive_2_y_joint = self.urdf_logger.joint_entity_path(passive_2_y)
        passive_2_z = self._get_joint("passive_2_z")
        passive_2_z_joint = self.urdf_logger.joint_entity_path(passive_2_z)

        passive_3_x = self._get_joint("passive_3_x")
        passive_3_x_joint = self.urdf_logger.joint_entity_path(passive_3_x)
        passive_3_y = self._get_joint("passive_3_y")
        passive_3_y_joint = self.urdf_logger.joint_entity_path(passive_3_y)
        passive_3_z = self._get_joint("passive_3_z")
        passive_3_z_joint = self.urdf_logger.joint_entity_path(passive_3_z)

        passive_4_x = self._get_joint("passive_4_x")
        passive_4_x_joint = self.urdf_logger.joint_entity_path(passive_4_x)
        passive_4_y = self._get_joint("passive_4_y")
        passive_4_y_joint = self.urdf_logger.joint_entity_path(passive_4_y)
        passive_4_z = self._get_joint("passive_4_z")
        passive_4_z_joint = self.urdf_logger.joint_entity_path(passive_4_z)

        passive_5_x = self._get_joint("passive_5_x")
        passive_5_x_joint = self.urdf_logger.joint_entity_path(passive_5_x)
        passive_5_y = self._get_joint("passive_5_y")
        passive_5_y_joint = self.urdf_logger.joint_entity_path(passive_5_y)
        passive_5_z = self._get_joint("passive_5_z")
        passive_5_z_joint = self.urdf_logger.joint_entity_path(passive_5_z)

        passive_6_x = self._get_joint("passive_6_x")
        passive_6_x_joint = self.urdf_logger.joint_entity_path(passive_6_x)
        passive_6_y = self._get_joint("passive_6_y")
        passive_6_y_joint = self.urdf_logger.joint_entity_path(passive_6_y)
        passive_6_z = self._get_joint("passive_6_z")
        passive_6_z_joint = self.urdf_logger.joint_entity_path(passive_6_z)

        passive_7_x = self._get_joint("passive_7_x")
        passive_7_x_joint = self.urdf_logger.joint_entity_path(passive_7_x)
        passive_7_y = self._get_joint("passive_7_y")
        passive_7_y_joint = self.urdf_logger.joint_entity_path(passive_7_y)
        passive_7_z = self._get_joint("passive_7_z")
        passive_7_z_joint = self.urdf_logger.joint_entity_path(passive_7_z)

        url = "http://localhost:8000/api/state/full"

        params = {
            "with_control_mode": "false",
            "with_head_pose": "false",
            "with_target_head_pose": "false",
            "with_head_joints": "true",
            "with_target_head_joints": "false",
            "with_body_yaw": "false",  # already in head_joints
            "with_target_body_yaw": "false",
            "with_antenna_positions": "true",
            "with_target_antenna_positions": "false",
            "use_pose_matrix": "false",
        }

        while not self.running.is_set():
            msg = requests.get(url, params=params)

            if msg.status_code != 200:
                self.logger.error(
                    f"Request failed with status {msg.status_code}: {msg.text}"
                )
                time.sleep(0.1)
                continue
            try:
                data = json.loads(msg.text)
                self.logger.debug(f"Data: {data}")
            except Exception:
                continue

            rr.set_time("reachymini", timestamp=time.time(), recording=self.recording)

            if "head_pose" in data:
                pose = data["head_pose"]
                print(
                    f"x={pose['x']}, y={pose['y']}, z={pose['z']}, "
                    f"roll={pose['roll']}, pitch={pose['pitch']}, yaw={pose['yaw']}"
                )
            if "antennas_positions" in data:
                antennas = data["antennas_positions"]
                if antennas is not None:
                    antenna_left.origin.rotation = [
                        -0.0581863,
                        -0.527253,
                        -0.0579647 + antennas[0],
                    ]
                    self.urdf_logger.log_joint(
                        antenna_left_joint,
                        joint=antenna_left,
                        recording=self.recording,
                    )
                    antenna_right.origin.rotation = [
                        1.5708,
                        -1.40009 - antennas[1],
                        -1.48353,
                    ]
                    self.urdf_logger.log_joint(
                        antenna_right_joint,
                        joint=antenna_right,
                        recording=self.recording,
                    )
            if "head_joints" in data:
                head_joints = data["head_joints"]
                motor_1.origin.rotation = [
                    -1.5708,
                    7.95539e-16 + head_joints[1],
                    2.0944,
                ]
                self.urdf_logger.log_joint(
                    motor_1_joint, joint=motor_1, recording=self.recording
                )

                motor_2.origin.rotation = [
                    1.5708,
                    -1.05471e-15 - head_joints[2],
                    -1.0472,
                ]
                self.urdf_logger.log_joint(
                    motor_2_joint, joint=motor_2, recording=self.recording
                )

                motor_3.origin.rotation = [
                    -1.5708,
                    -2.77556e-16 + head_joints[3],
                    3.14916e-16,
                ]
                self.urdf_logger.log_joint(
                    motor_3_joint, joint=motor_3, recording=self.recording
                )
                motor_4.origin.rotation = [
                    1.5708,
                    4.44377e-15 - head_joints[4],
                    3.14159,
                ]
                self.urdf_logger.log_joint(
                    motor_4_joint, joint=motor_4, recording=self.recording
                )
                motor_5.origin.rotation = [
                    -1.5708,
                    4.85252e-16 + head_joints[5],
                    -2.0944,
                ]
                self.urdf_logger.log_joint(
                    motor_5_joint, joint=motor_5, recording=self.recording
                )
                motor_6.origin.rotation = [
                    1.5708,
                    1.05471e-15 - head_joints[6],
                    1.0472,
                ]
                self.urdf_logger.log_joint(
                    motor_6_joint, joint=motor_6, recording=self.recording
                )

                motor_yaw.origin.rotation = [
                    3.14159,
                    -2.22045e-16,
                    1.5708 - head_joints[0],
                ]
                self.urdf_logger.log_joint(
                    motor_yaw_joint, joint=motor_yaw, recording=self.recording
                )
            time.sleep(0.1)

    def log_rerun_info(self):
        """Log the Rerun application and recording IDs."""
        while not self.running.is_set():
            self.backend.rerun_ids_publisher.put(
                json.dumps(
                    {
                        "recording_id": self.recording.get_recording_id(),
                        "application_id": self.app_id,
                    }
                )
            )
            time.sleep(1)

    def log_movement(self):
        """Log the movement data to Rerun."""
        antenna_left = self._get_joint("left_antenna")
        antenna_left_joint = self.urdf_logger.joint_entity_path(antenna_left)
        antenna_right = self._get_joint("right_antenna")
        antenna_right_joint = self.urdf_logger.joint_entity_path(antenna_right)

        motor_1 = self._get_joint("1")
        motor_1_joint = self.urdf_logger.joint_entity_path(motor_1)
        motor_2 = self._get_joint("2")
        motor_2_joint = self.urdf_logger.joint_entity_path(motor_2)
        motor_3 = self._get_joint("3")
        motor_3_joint = self.urdf_logger.joint_entity_path(motor_3)
        motor_4 = self._get_joint("4")
        motor_4_joint = self.urdf_logger.joint_entity_path(motor_4)
        motor_5 = self._get_joint("5")
        motor_5_joint = self.urdf_logger.joint_entity_path(motor_5)
        motor_6 = self._get_joint("6")
        motor_6_joint = self.urdf_logger.joint_entity_path(motor_6)
        motor_yaw = self._get_joint("all_yaw")
        motor_yaw_joint = self.urdf_logger.joint_entity_path(motor_yaw)

        passive_1_x = self._get_joint("passive_1_x")
        passive_1_x_joint = self.urdf_logger.joint_entity_path(passive_1_x)
        passive_1_y = self._get_joint("passive_1_y")
        passive_1_y_joint = self.urdf_logger.joint_entity_path(passive_1_y)
        passive_1_z = self._get_joint("passive_1_z")
        passive_1_z_joint = self.urdf_logger.joint_entity_path(passive_1_z)

        passive_2_x = self._get_joint("passive_2_x")
        passive_2_x_joint = self.urdf_logger.joint_entity_path(passive_2_x)
        passive_2_y = self._get_joint("passive_2_y")
        passive_2_y_joint = self.urdf_logger.joint_entity_path(passive_2_y)
        passive_2_z = self._get_joint("passive_2_z")
        passive_2_z_joint = self.urdf_logger.joint_entity_path(passive_2_z)

        passive_3_x = self._get_joint("passive_3_x")
        passive_3_x_joint = self.urdf_logger.joint_entity_path(passive_3_x)
        passive_3_y = self._get_joint("passive_3_y")
        passive_3_y_joint = self.urdf_logger.joint_entity_path(passive_3_y)
        passive_3_z = self._get_joint("passive_3_z")
        passive_3_z_joint = self.urdf_logger.joint_entity_path(passive_3_z)

        passive_4_x = self._get_joint("passive_4_x")
        passive_4_x_joint = self.urdf_logger.joint_entity_path(passive_4_x)
        passive_4_y = self._get_joint("passive_4_y")
        passive_4_y_joint = self.urdf_logger.joint_entity_path(passive_4_y)
        passive_4_z = self._get_joint("passive_4_z")
        passive_4_z_joint = self.urdf_logger.joint_entity_path(passive_4_z)

        passive_5_x = self._get_joint("passive_5_x")
        passive_5_x_joint = self.urdf_logger.joint_entity_path(passive_5_x)
        passive_5_y = self._get_joint("passive_5_y")
        passive_5_y_joint = self.urdf_logger.joint_entity_path(passive_5_y)
        passive_5_z = self._get_joint("passive_5_z")
        passive_5_z_joint = self.urdf_logger.joint_entity_path(passive_5_z)

        passive_6_x = self._get_joint("passive_6_x")
        passive_6_x_joint = self.urdf_logger.joint_entity_path(passive_6_x)
        passive_6_y = self._get_joint("passive_6_y")
        passive_6_y_joint = self.urdf_logger.joint_entity_path(passive_6_y)
        passive_6_z = self._get_joint("passive_6_z")
        passive_6_z_joint = self.urdf_logger.joint_entity_path(passive_6_z)

        passive_7_x = self._get_joint("passive_7_x")
        passive_7_x_joint = self.urdf_logger.joint_entity_path(passive_7_x)
        passive_7_y = self._get_joint("passive_7_y")
        passive_7_y_joint = self.urdf_logger.joint_entity_path(passive_7_y)
        passive_7_z = self._get_joint("passive_7_z")
        passive_7_z_joint = self.urdf_logger.joint_entity_path(passive_7_z)

        if self.cam is not None:
            ret, frame = self.cam.read()
            if ret:
                cam_name = self._get_joint("camera_optical_frame")
                cam_joint = self.urdf_logger.joint_entity_path(cam_name)

                K = np.array(
                    [
                        [550.3564, 0.0, 638.0112],
                        [0.0, 549.1653, 364.589],
                        [0.0, 0.0, 1.0],
                    ]
                )

        while not self.running.is_set():
            rr.set_time("deamon", timestamp=time.time(), recording=self.recording)

            # hard coded numbers are from the URDF file
            antennas = self.backend.get_antenna_joint_positions()
            if antennas is not None:
                antenna_left.origin.rotation = [
                    -0.0581863,
                    -0.527253,
                    -0.0579647 + antennas[0],
                ]
                self.urdf_logger.log_joint(
                    antenna_left_joint, joint=antenna_left, recording=self.recording
                )
                antenna_right.origin.rotation = [
                    1.5708,
                    -1.40009 - antennas[1],
                    -1.48353,
                ]
                self.urdf_logger.log_joint(
                    antenna_right_joint, joint=antenna_right, recording=self.recording
                )

            head = self.backend.get_head_joint_positions()
            if head is not None:
                motor_1.origin.rotation = [
                    -1.5708,
                    7.95539e-16 + head[1],
                    2.0944,
                ]
                self.urdf_logger.log_joint(
                    motor_1_joint, joint=motor_1, recording=self.recording
                )

                motor_2.origin.rotation = [
                    1.5708,
                    -1.05471e-15 - head[2],
                    -1.0472,
                ]
                self.urdf_logger.log_joint(
                    motor_2_joint, joint=motor_2, recording=self.recording
                )

                motor_3.origin.rotation = [
                    -1.5708,
                    -2.77556e-16 + head[3],
                    3.14916e-16,
                ]
                self.urdf_logger.log_joint(
                    motor_3_joint, joint=motor_3, recording=self.recording
                )
                motor_4.origin.rotation = [
                    1.5708,
                    4.44377e-15 - head[4],
                    3.14159,
                ]
                self.urdf_logger.log_joint(
                    motor_4_joint, joint=motor_4, recording=self.recording
                )
                motor_5.origin.rotation = [
                    -1.5708,
                    4.85252e-16 + head[5],
                    -2.0944,
                ]
                self.urdf_logger.log_joint(
                    motor_5_joint, joint=motor_5, recording=self.recording
                )
                motor_6.origin.rotation = [
                    1.5708,
                    1.05471e-15 - head[6],
                    1.0472,
                ]
                self.urdf_logger.log_joint(
                    motor_6_joint, joint=motor_6, recording=self.recording
                )

                motor_yaw.origin.rotation = [3.14159, -2.22045e-16, 1.5708 - head[0]]
                self.urdf_logger.log_joint(
                    motor_yaw_joint, joint=motor_yaw, recording=self.recording
                )

                # updating kinematic chains with placo
                self.head_kinematics.fk(head)

                self._set_rod_rotation(
                    "passive_1_x",
                    passive_1_x,
                    passive_1_x_joint,
                    [-3.09302, 0.0258157, -1.08824],
                    0,
                )
                self._set_rod_rotation(
                    "passive_1_y",
                    passive_1_y,
                    passive_1_y_joint,
                    [6.20901e-17, 1.31226e-17, -3.8231e-17],
                    1,
                )
                self._set_rod_rotation(
                    "passive_1_z",
                    passive_1_z,
                    passive_1_z_joint,
                    [6.20901e-17, 1.31226e-17, -3.8231e-17],
                    2,
                )

                self._set_rod_rotation(
                    "passive_2_x",
                    passive_2_x,
                    passive_2_x_joint,
                    [-3.09518, 0.0258157, 1.08824],
                    0,
                )

                self._set_rod_rotation(
                    "passive_2_y",
                    passive_2_y,
                    passive_2_y_joint,
                    [1.03529e-16, -2.76081e-18, 1.06701e-17],
                    1,
                )
                self._set_rod_rotation(
                    "passive_2_z",
                    passive_2_z,
                    passive_2_z_joint,
                    [1.03529e-16, -2.76081e-18, 1.06701e-17],
                    2,
                )

                self._set_rod_rotation(
                    "passive_3_x",
                    passive_3_x,
                    passive_3_x_joint,
                    [-0.0802622, 0.0258157, -1.08824],
                    0,
                )
                self._set_rod_rotation(
                    "passive_3_y",
                    passive_3_y,
                    passive_3_y_joint,
                    [-1.00225e-17, 4.94486e-18, 1.68244e-17],
                    1,
                )
                self._set_rod_rotation(
                    "passive_3_z",
                    passive_3_z,
                    passive_3_z_joint,
                    [-1.00225e-17, 4.94486e-18, 1.68244e-17],
                    2,
                )

                self._set_rod_rotation(
                    "passive_4_x",
                    passive_4_x,
                    passive_4_x_joint,
                    [0.118327, 0.0258157, 1.08824],
                    0,
                )

                self._set_rod_rotation(
                    "passive_4_y",
                    passive_4_y,
                    passive_4_y_joint,
                    [-5.32364e-18, 3.35739e-18, 2.46433e-17],
                    1,
                )

                self._set_rod_rotation(
                    "passive_4_z",
                    passive_4_z,
                    passive_4_z_joint,
                    [-5.32364e-18, 3.35739e-18, 2.46433e-17],
                    2,
                )

                self._set_rod_rotation(
                    "passive_5_x",
                    passive_5_x,
                    passive_5_x_joint,
                    [-2.94446, 0.0258157, -1.08824],
                    0,
                )

                self._set_rod_rotation(
                    "passive_5_y",
                    passive_5_y,
                    passive_5_y_joint,
                    [-3.08493e-17, 2.79333e-17, 5.24089e-17],
                    1,
                )

                self._set_rod_rotation(
                    "passive_5_z",
                    passive_5_z,
                    passive_5_z_joint,
                    [-3.08493e-17, 2.79333e-17, 5.24089e-17],
                    2,
                )

                self._set_rod_rotation(
                    "passive_6_x",
                    passive_6_x,
                    passive_6_x_joint,
                    [-3.09524, 0.0258157, 1.08824],
                    0,
                )

                self._set_rod_rotation(
                    "passive_6_y",
                    passive_6_y,
                    passive_6_y_joint,
                    [9.77399e-17, 3.28422e-18, 1.01949e-16],
                    1,
                )

                self._set_rod_rotation(
                    "passive_6_z",
                    passive_6_z,
                    passive_6_z_joint,
                    [9.77399e-17, 3.28422e-18, 1.01949e-16],
                    2,
                )

                self._set_rod_rotation(
                    "passive_7_x",
                    passive_7_x,
                    passive_7_x_joint,
                    [-3.14025, 0.0530311, 1.08768],
                    0,
                )

                self._set_rod_rotation(
                    "passive_7_y",
                    passive_7_y,
                    passive_7_y_joint,
                    [-1.00928e-30, 6.45118e-17, -2.11395e-31],
                    1,
                )

                self._set_rod_rotation(
                    "passive_7_z",
                    passive_7_z,
                    passive_7_z_joint,
                    [-1.00928e-30, 6.45118e-17, -2.11395e-31],
                    2,
                )

            if self.cam is not None:
                cam_name.origin.rotation = [3.14159, 1.0472, 3.14159]
                cam_name.origin.position = [0.0244171, -0.0524, 0.0147383]
                self.urdf_logger.log_joint(
                    cam_joint, joint=cam_name, recording=self.recording
                )

                ret, frame = self.cam.read()

                if ret:
                    rr.log(
                        f"{cam_joint}/image",
                        rr.Pinhole(
                            image_from_camera=rr.datatypes.Mat3x3(K),
                            width=frame.shape[1],
                            height=frame.shape[0],
                            image_plane_distance=0.8,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        ),
                    )

                    # ToDo: this is suboptimal since the camera outputs a MJPEG stream
                    # use alternative to opencv
                    ret, encoded_image = cv2.imencode(".jpg", frame)
                    if ret:
                        rr.log(
                            f"{cam_joint}/image",
                            rr.EncodedImage(
                                contents=encoded_image, media_type="image/jpeg"
                            ),
                        )
                    else:
                        self.logger.error("Failed to encode frame to JPEG.")

                else:
                    self.logger.error("Failed to grab frame from camera.")

            time.sleep(0.1)
