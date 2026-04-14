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
from typing import Dict, Optional

import numpy as np
import requests
import rerun as rr
from scipy.spatial.transform import Rotation as R

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
        status = self._reachymini.client.get_status()
        if status.wireless_version and status.wlan_ip:
            self._robot_ip = status.wlan_ip

        self.recording = rr.get_global_data_recording()

        script_dir = os.path.dirname(os.path.abspath(__file__))

        urdf_path = os.path.join(
            script_dir, "../descriptions/reachy_mini/urdf/robot_no_collision.urdf"
        )
        asset_path = os.path.join(script_dir, "../descriptions/reachy_mini/urdf")

        fixed_urdf = self.set_absolute_path_to_urdf(urdf_path, asset_path)
        self.logger.debug(
            f"Using URDF file: {fixed_urdf} with absolute paths for Rerun."
        )

        self.head_kinematics = PlacoKinematics(fixed_urdf)

        # Load URDF tree for joint metadata (frame names, origins, axes)
        self._urdf_tree = rr.urdf.UrdfTree.from_file_path(fixed_urdf)
        self._joints_by_name: Dict[str, rr.urdf.UrdfJoint] = {
            joint.name: joint for joint in self._urdf_tree.joints()
        }

        rr.set_time("reachymini", timestamp=time.time(), recording=self.recording)

        # Use the native URDF loader in Rerun to visualize Reachy Mini's model
        # Logging as non-static to allow updating joint positions
        rr.log_file_from_path(
            fixed_urdf,
            static=False,
            entity_path_prefix="ReachyMini",
            recording=self.recording,
        )

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

    def _log_joint_transform(
        self,
        joint_name: str,
        angle: float,
        axis: list[float] | None = None,
    ) -> None:
        """Log a joint transform to Rerun using frame-based transforms.

        Args:
            joint_name: Name of the joint in the URDF.
            angle: Joint angle in radians.
            axis: Optional axis override. If None, uses the URDF joint axis
                  via compute_transform.

        """
        joint = self._joints_by_name[joint_name]

        if axis is None:
            # Use compute_transform for joints with correct URDF axes
            transform = joint.compute_transform(angle)
            rr.log(f"transforms/{joint_name}", transform, recording=self.recording)
        else:
            # Manual transform for joints with axis overrides.
            # The URDF axis definitions don't match the real axis of rotation
            # for some joints (e.g. stewart platform, body yaw), so we override.
            base_rpy = list(joint.origin_rpy or [0.0, 0.0, 0.0])
            effective_axis = np.array(axis)
            target_euler = np.array(base_rpy) + (effective_axis * angle)
            target_translation = list(joint.origin_xyz or [0.0, 0.0, 0.0])

            rr.log(
                f"transforms/{joint_name}",
                rr.Transform3D(
                    translation=target_translation,
                    quaternion=R.from_euler("xyz", target_euler).as_quat(),
                    parent_frame=joint.parent_link,
                    child_frame=joint.child_link,
                ),
                recording=self.recording,
            )

    def log_camera(self) -> None:
        """Log the camera image to Rerun."""
        if self._reachymini.media.camera is None:
            self.logger.warning("Camera is not initialized.")
            return

        self.logger.info("Starting camera logging to Rerun.")

        # Connect the camera entity to the camera frame from the URDF
        cam_joint = self._joints_by_name.get("camera_optical_frame")
        cam_frame = cam_joint.child_link if cam_joint else "camera_optical_frame"
        rr.log(
            "camera",
            rr.Transform3D(parent_frame=cam_frame),
            recording=self.recording,
        )

        while not self.running.is_set():
            rr.set_time("reachymini", timestamp=time.time(), recording=self.recording)
            frame = self._reachymini.media.get_frame()
            if frame is not None:
                if isinstance(frame, bytes):
                    self.logger.warning(
                        "Received frame is jpeg. Please use default backend."
                    )
                    return

            else:
                return

            K = np.array(
                [
                    [550.3564, 0.0, 638.0112],
                    [0.0, 549.1653, 364.589],
                    [0.0, 0.0, 1.0],
                ]
            )

            rr.log(
                "camera/image",
                rr.Pinhole(
                    image_from_camera=rr.datatypes.Mat3x3(K),
                    width=frame.shape[1],
                    height=frame.shape[0],
                    image_plane_distance=0.8,
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
                rr.Image(frame, color_model="bgr").compress(),
                recording=self.recording,
            )

            time.sleep(0.03)  # ~30fps

    def log_movements(self) -> None:
        """Log the movement data to Rerun."""
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

            if "antennas_position" in data and data["antennas_position"] is not None:
                antennas = data["antennas_position"]
                if antennas is not None:
                    self._log_joint_transform("left_antenna", antennas[0])
                    self._log_joint_transform("right_antenna", antennas[1])

            if "head_joints" in data and data["head_joints"] is not None:
                head_joints = data["head_joints"]

                # The joint axis definitions in the URDF do not match the real axis of rotation
                # due to URDF not supporting ball joints properly.
                self._log_joint_transform("yaw_body", -head_joints[0], axis=[0, 0, 1])
                self._log_joint_transform("stewart_1", -head_joints[1], axis=[0, 1, 0])
                self._log_joint_transform("stewart_2", head_joints[2], axis=[0, 1, 0])
                self._log_joint_transform("stewart_3", -head_joints[3], axis=[0, 1, 0])
                self._log_joint_transform("stewart_4", head_joints[4], axis=[0, 1, 0])
                self._log_joint_transform("stewart_5", -head_joints[5], axis=[0, 1, 0])
                self._log_joint_transform("stewart_6", head_joints[6], axis=[0, 1, 0])

            if "passive_joints" in data and data["passive_joints"] is not None:
                passive_joints = data["passive_joints"]

                for axis_idx, axis in enumerate(["x", "y", "z"]):
                    for i in range(1, 8):
                        joint_name = f"passive_{i}_{axis}"
                        value_index = (i - 1) * 3 + axis_idx

                        override_axis = [0.0, 0.0, 0.0]
                        override_axis[axis_idx] = 1.0
                        self._log_joint_transform(
                            joint_name,
                            passive_joints[value_index],
                            axis=override_axis,
                        )

            time.sleep(0.1)
