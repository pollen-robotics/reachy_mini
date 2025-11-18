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
from typing import List, Optional

import cv2
import numpy as np
import requests
import rerun as rr

try:
    from rerun_loader_urdf import URDFLogger
except ImportError:
    raise ImportError(
        "The 'rerun-loader-urdf' package is required for this module. "
        "Please install it from the GitHub repository: "
        "pip install git+https://github.com/rerun-io/rerun-loader-python-example-urdf.git"
    )
from urdf_parser_py import urdf

from reachy_mini.kinematics.placo_kinematics import PlacoKinematics
from reachy_mini.media.media_manager import MediaBackend
from reachy_mini.reachy_mini import ReachyMini


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
            reachymini (ReachyMini): The Reachy Mini instance to log.
            app_id (str): The application ID for Rerun. Defaults to reachy_mini_daemon.
            spawn (bool): If True, spawn the Rerun server. Defaults to True.

        """
        rr.init(app_id, spawn=spawn)
        self.app_id = app_id
        self._reachymini = reachymini
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(reachymini.logger.getEffectiveLevel())

        self._robot_ip = "localhost"
        if self._reachymini.client.get_status()["wireless_version"]:
            self._robot_ip = self._reachymini.client.get_status()["wlan_ip"]

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
        self.thread_log_camera: Optional[Thread] = None
        if (
            reachymini.media.backend == MediaBackend.GSTREAMER
            or reachymini.media.backend == MediaBackend.DEFAULT
        ):
            self.thread_log_camera = Thread(target=self.log_camera, daemon=True)
        self.thread_log_movements = Thread(target=self.log_movements, daemon=True)

    def set_absolute_path_to_urdf(self, urdf_path: str, abs_path: str) -> str:
        """Set the absolute paths in the URDF file. Rerun cannot read the "package://" paths."""
        with open(urdf_path, "r") as f:
            urdf_content = f.read()
        urdf_content_mod = urdf_content.replace("package://", f"file://{abs_path}/")

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".urdf") as tmp_file:
            tmp_file.write(urdf_content_mod)
            return tmp_file.name

    def start(self) -> None:
        """Start the Rerun logging thread."""
        if self.thread_log_camera is not None:
            self.thread_log_camera.start()
        self.thread_log_movements.start()

    def stop(self) -> None:
        """Stop the Rerun logging thread."""
        self.running.set()

    def _get_joint(self, joint_name: str) -> urdf.Joint:
        for j in self.urdf_logger.urdf.joints:
            if j.name == joint_name:
                return j
        raise RuntimeError("Invalid joint name")

    def _set_rod_rotation(
        self,
        joint_rot: float,
        joint: urdf.Joint,
        joint_path: str,
        urdf_offset: List[float],
        id_rotation: int,
    ) -> None:
        urdf_offset[id_rotation] += joint_rot
        joint.origin.rotation = urdf_offset

        self.urdf_logger.log_joint(joint_path, joint=joint, recording=self.recording)

    def log_camera(self) -> None:
        """Log the camera image to Rerun."""
        if self._reachymini.media.camera is None:
            self.logger.warning("Camera is not initialized.")
            return

        self.logger.info("Starting camera logging to Rerun.")

        while not self.running.is_set():
            rr.set_time("reachymini", timestamp=time.time(), recording=self.recording)
            frame = self._reachymini.media.get_frame()
            if frame is not None:
                if isinstance(frame, bytes):
                    self.logger.warning(
                        "Received frame is jpeg. Please use default backend."
                    )
                    return

                cam_name = self._get_joint("camera_optical_frame")
                cam_joint = self.urdf_logger.joint_entity_path(cam_name)
            else:
                return

            K = np.array(
                [
                    [550.3564, 0.0, 638.0112],
                    [0.0, 549.1653, 364.589],
                    [0.0, 0.0, 1.0],
                ]
            )

            # camera optical frame from URDF
            cam_name.origin.rotation = [2.92045, 1.01151, 1.9102]
            cam_name.origin.position = [-0.0321159, -0.05047, 0.00257878]
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

            ret, encoded_image = cv2.imencode(".jpg", frame)
            if ret:
                rr.log(
                    f"{cam_joint}/image",
                    rr.EncodedImage(contents=encoded_image, media_type="image/jpeg"),
                )
            else:
                self.logger.error("Failed to encode frame to JPEG.")

            time.sleep(0.3)  # ~30fps

    def log_movements(self) -> None:
        """Log the movement data to Rerun."""
        antenna_left = self._get_joint("left_antenna")
        antenna_left_joint = self.urdf_logger.joint_entity_path(antenna_left)
        antenna_right = self._get_joint("right_antenna")
        antenna_right_joint = self.urdf_logger.joint_entity_path(antenna_right)

        motor_1 = self._get_joint("stewart_1")
        motor_1_joint = self.urdf_logger.joint_entity_path(motor_1)
        motor_2 = self._get_joint("stewart_2")
        motor_2_joint = self.urdf_logger.joint_entity_path(motor_2)
        motor_3 = self._get_joint("stewart_3")
        motor_3_joint = self.urdf_logger.joint_entity_path(motor_3)
        motor_4 = self._get_joint("stewart_4")
        motor_4_joint = self.urdf_logger.joint_entity_path(motor_4)
        motor_5 = self._get_joint("stewart_5")
        motor_5_joint = self.urdf_logger.joint_entity_path(motor_5)
        motor_6 = self._get_joint("stewart_6")
        motor_6_joint = self.urdf_logger.joint_entity_path(motor_6)
        motor_yaw = self._get_joint("yaw_body")
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

        url = f"http://{self._robot_ip}:8000/api/state/full"

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
            "with_passive_joints": "true",
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

            # hardcoded offsets are from the URDF file
            if "antennas_position" in data and data["antennas_position"] is not None:
                antennas = data["antennas_position"]
                if antennas is not None:
                    antenna_left.origin.rotation = [
                        2.93649,
                        0.508471,
                        2.10225 + antennas[0],
                    ]
                    self.urdf_logger.log_joint(
                        antenna_left_joint,
                        joint=antenna_left,
                        recording=self.recording,
                    )
                    antenna_right.origin.rotation = [
                        1.63922,
                        1.39152 + antennas[1],
                        0.701924,
                    ]
                    self.urdf_logger.log_joint(
                        antenna_right_joint,
                        joint=antenna_right,
                        recording=self.recording,
                    )
            if "head_joints" in data and data["head_joints"] is not None:
                head_joints = data["head_joints"]
                motor_1.origin.rotation = [
                    1.5708,
                    5.91241e-14 - head_joints[1],
                    1.0472,
                ]
                self.urdf_logger.log_joint(
                    motor_1_joint, joint=motor_1, recording=self.recording
                )

                motor_2.origin.rotation = [
                    -1.5708,
                    1.47282e-13 + head_joints[2],
                    -2.0944,
                ]
                self.urdf_logger.log_joint(
                    motor_2_joint, joint=motor_2, recording=self.recording
                )

                motor_3.origin.rotation = [
                    1.5708,
                    5.72146e-14 - head_joints[3],
                    3.14159,
                ]
                self.urdf_logger.log_joint(
                    motor_3_joint, joint=motor_3, recording=self.recording
                )
                motor_4.origin.rotation = [
                    -1.5708,
                    -7.49452e-14 + head_joints[4],
                    3.47181e-14,
                ]
                self.urdf_logger.log_joint(
                    motor_4_joint, joint=motor_4, recording=self.recording
                )
                motor_5.origin.rotation = [
                    1.5708,
                    -1.79054e-13 - head_joints[5],
                    -1.0472,
                ]
                self.urdf_logger.log_joint(
                    motor_5_joint, joint=motor_5, recording=self.recording
                )
                motor_6.origin.rotation = [
                    -1.5708,
                    -5.32144e-14 + head_joints[6],
                    2.0944,
                ]
                self.urdf_logger.log_joint(
                    motor_6_joint, joint=motor_6, recording=self.recording
                )

                motor_yaw.origin.rotation = [
                    -3.74039e-16,
                    1.77636e-15,
                    1.5708 - head_joints[0],
                ]
                self.urdf_logger.log_joint(
                    motor_yaw_joint, joint=motor_yaw, recording=self.recording
                )

            if "passive_joints" in data and data["passive_joints"] is not None:
                passive_joints = data["passive_joints"]
                self._set_rod_rotation(
                    passive_joints[0],  # "passive_1_x",
                    passive_1_x,
                    passive_1_x_joint,
                    [-0.13754, -0.0882156, 2.10349],
                    0,
                )
                self._set_rod_rotation(
                    passive_joints[1],  # "passive_1_y",
                    passive_1_y,
                    passive_1_y_joint,
                    [-4.80812e-17, 3.69195e-17, -3.11194e-17],
                    1,
                )
                self._set_rod_rotation(
                    passive_joints[2],  # "passive_1_z",
                    passive_1_z,
                    passive_1_z_joint,
                    [-4.80812e-17, 3.69195e-17, -3.11194e-17],
                    2,
                )

                self._set_rod_rotation(
                    passive_joints[3],  # "passive_2_x",
                    passive_2_x,
                    passive_2_x_joint,
                    [-3.14159, 5.37396e-16, -3.14159],
                    0,
                )

                self._set_rod_rotation(
                    passive_joints[4],  # "passive_2_y",
                    passive_2_y,
                    passive_2_y_joint,
                    [-4.29816e-30, 7.32263e-17, -8.42229e-30],
                    1,
                )
                self._set_rod_rotation(
                    passive_joints[5],  # "passive_2_z",
                    passive_2_z,
                    passive_2_z_joint,
                    [-4.29816e-30, 7.32263e-17, -8.42229e-30],
                    2,
                )

                self._set_rod_rotation(
                    passive_joints[6],  # "passive_3_x",
                    passive_3_x,
                    passive_3_x_joint,
                    [0.373569, 0.0882156, -1.0381],
                    0,
                )
                self._set_rod_rotation(
                    passive_joints[7],  # "passive_3_y",
                    passive_3_y,
                    passive_3_y_joint,
                    [-4.71809e-17, -7.61919e-18, 5.54539e-17],
                    1,
                )
                self._set_rod_rotation(
                    passive_joints[8],  # "passive_3_z",
                    passive_3_z,
                    passive_3_z_joint,
                    [-4.71809e-17, -7.61919e-18, 5.54539e-17],
                    2,
                )

                self._set_rod_rotation(
                    passive_joints[9],  # "passive_4_x",
                    passive_4_x,
                    passive_4_x_joint,
                    [-0.0860846, 0.0882156, 1.0381],
                    0,
                )

                self._set_rod_rotation(
                    passive_joints[10],  # "passive_4_y",
                    passive_4_y,
                    passive_4_y_joint,
                    [-3.54762e-18, 2.76754e-18, -1.40989e-17],
                    1,
                )

                self._set_rod_rotation(
                    passive_joints[11],  # "passive_4_z",
                    passive_4_z,
                    passive_4_z_joint,
                    [-3.54762e-18, 2.76754e-18, -1.40989e-17],
                    2,
                )

                self._set_rod_rotation(
                    passive_joints[12],  # "passive_5_x",
                    passive_5_x,
                    passive_5_x_joint,
                    [0.123977, 0.0882156, -1.0381],
                    0,
                )

                self._set_rod_rotation(
                    passive_joints[13],  # "passive_5_y",
                    passive_5_y,
                    passive_5_y_joint,
                    [-1.10888e-16, -3.21444e-17, 3.55033e-17],
                    1,
                )

                self._set_rod_rotation(
                    passive_joints[14],  # "passive_5_z",
                    passive_5_z,
                    passive_5_z_joint,
                    [-1.10888e-16, -3.21444e-17, 3.55033e-17],
                    2,
                )

                self._set_rod_rotation(
                    passive_joints[15],  # "passive_6_x",
                    passive_6_x,
                    passive_6_x_joint,
                    [3.0613, 0.0882156, 1.0381],
                    0,
                )

                self._set_rod_rotation(
                    passive_joints[16],  # "passive_6_y",
                    passive_6_y,
                    passive_6_y_joint,
                    [-7.27418e-17, 2.10388e-17, 4.15523e-17],
                    1,
                )

                self._set_rod_rotation(
                    passive_joints[17],  # "passive_6_z",
                    passive_6_z,
                    passive_6_z_joint,
                    [-7.27418e-17, 2.10388e-17, 4.15523e-17],
                    2,
                )

                self._set_rod_rotation(
                    passive_joints[18],  # "passive_7_x",
                    passive_7_x,
                    passive_7_x_joint,
                    [3.14159, 2.10388e-17, 4.15523e-17],
                    0,
                )

                self._set_rod_rotation(
                    passive_joints[19],  # "passive_7_y",
                    passive_7_y,
                    passive_7_y_joint,
                    [-7.27418e-17, -2.10388e-17, -4.15523e-17],
                    1,
                )

                self._set_rod_rotation(
                    passive_joints[20],  # "passive_7_z",
                    passive_7_z,
                    passive_7_z_joint,
                    [-7.27418e-17, -2.10388e-17, -4.15523e-17],
                    2,
                )

            time.sleep(0.1)
