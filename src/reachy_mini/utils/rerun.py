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
from datetime import datetime
from threading import Event, Thread
from typing import Dict, Optional

import numpy as np
import requests
import rerun as rr

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
        rr.log_file_from_path(
            fixed_urdf,
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
        """Start the Rerun logging threads."""
        if self.thread_log_camera is not None:
            self.thread_log_camera.start()
        self.thread_log_movements.start()

    def stop(self) -> None:
        """Stop the Rerun logging threads."""
        self.running.set()

    def _log_all_joint_transforms(self) -> None:
        """Log all joint transforms from the placo FK robot state to Rerun.

        The Stewart platform is a parallel mechanism that URDF represents as a
        tree. Only one branch connects to the head, so all passive joint values
        must be solved via constrained FK before logging. After placo FK solves,
        we read each joint's angle and use compute_transform to log it.
        """
        for joint in self._urdf_tree.joints():
            if joint.joint_type != "revolute":
                continue
            angle = self.head_kinematics.robot.get_joint(joint.name)
            transform = joint.compute_transform(angle)
            rr.log(f"transforms/{joint.name}", transform, recording=self.recording)

    def log_camera(self) -> None:
        """Log the camera image to Rerun."""
        if self._reachymini.media.camera is None:
            self.logger.warning("Camera is not initialized.")
            return

        self.logger.info("Starting camera logging to Rerun.")

        # Connect the camera entity to the camera_optical frame from the URDF.
        # Logged as static since this is a fixed joint that doesn't change.
        rr.log(
            "camera",
            rr.Transform3D(parent_frame="camera_optical"),
            static=True,
            recording=self.recording,
        )

        cam_K = np.array(
            [
                [550.3564, 0.0, 638.0112],
                [0.0, 549.1653, 364.589],
                [0.0, 0.0, 1.0],
            ]
        )

        while not self.running.is_set():
            frame = self._reachymini.media.get_frame()
            if frame is not None:
                if isinstance(frame, bytes):
                    self.logger.warning(
                        "Received frame is jpeg. Please use default backend."
                    )
                    return
            else:
                return

            # TODO: Real timestamps exist in the pipeline (GStreamer buf.pts,
            # MuJoCo self.data.time) but camera.read() only returns raw pixels.
            # Using wall-clock time as a proxy. To fix properly, extend
            # camera read() to return (frame, timestamp) tuples.
            rr.set_time("reachymini", timestamp=time.time(), recording=self.recording)

            rr.log(
                "camera/image",
                rr.Pinhole(
                    image_from_camera=rr.datatypes.Mat3x3(cam_K),
                    width=frame.shape[1],
                    height=frame.shape[0],
                    image_plane_distance=0.8,
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
                rr.Image(frame, color_model="bgr").compress(),
                recording=self.recording,
            )
            # cap.read() blocks until next frame, no sleep needed

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
            "with_passive_joints": "false",  # computed via placo FK
        }

        # Names of the actuated joints (with servos)
        actuated_joint_names = [
            "yaw_body",
            "stewart_1", "stewart_2", "stewart_3",
            "stewart_4", "stewart_5", "stewart_6",
        ]

        target_period = 0.02  # 50Hz — matches daemon control loop rate
        session = requests.Session()

        while not self.running.is_set():
            loop_start = time.time()

            try:
                msg = session.get(url, params=params, timeout=0.5)
            except requests.RequestException:
                time.sleep(target_period)
                continue

            if msg.status_code != 200:
                self.logger.error(
                    f"Request failed with status {msg.status_code}: {msg.text}"
                )
                time.sleep(target_period)
                continue
            try:
                data = json.loads(msg.text)
            except Exception:
                continue

            # Use the API's own timestamp for accurate timing
            if "timestamp" in data and data["timestamp"] is not None:
                api_ts = datetime.fromisoformat(
                    data["timestamp"].replace("Z", "+00:00")
                ).timestamp()
            else:
                api_ts = time.time()
            rr.set_time("reachymini", timestamp=api_ts, recording=self.recording)

            if "head_joints" in data and data["head_joints"] is not None:
                head_joints = data["head_joints"]

                # Use placo FK to solve the full Stewart platform kinematic chain
                # (including all passive joints and closing constraints)
                joint_angles = np.array(head_joints)
                self.head_kinematics.fk(joint_angles, no_iterations=1)
                self._log_all_joint_transforms()

                # Log actuated joint positions as time-series
                for name, value in zip(actuated_joint_names, head_joints):
                    rr.log(f"joints/{name}", rr.Scalars(value), recording=self.recording)

            if "antennas_position" in data and data["antennas_position"] is not None:
                antennas = data["antennas_position"]
                if antennas is not None:
                    left = self._joints_by_name["left_antenna"]
                    right = self._joints_by_name["right_antenna"]
                    rr.log(
                        "transforms/left_antenna",
                        left.compute_transform(antennas[0]),
                        recording=self.recording,
                    )
                    rr.log(
                        "transforms/right_antenna",
                        right.compute_transform(antennas[1]),
                        recording=self.recording,
                    )
                    rr.log("joints/left_antenna", rr.Scalars(antennas[0]), recording=self.recording)
                    rr.log("joints/right_antenna", rr.Scalars(antennas[1]), recording=self.recording)

            # Sleep only the remainder to hit 50Hz target
            elapsed = time.time() - loop_start
            remaining = target_period - elapsed
            if remaining > 0:
                time.sleep(remaining)
