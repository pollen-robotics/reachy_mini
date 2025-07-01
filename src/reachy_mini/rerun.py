import os
import tempfile
import time
from threading import Event, Thread

import rerun as rr
from rerun_loader_urdf import URDFLogger
from urdf_parser_py import urdf

from reachy_mini.io import Backend


class Rerun:
    def __init__(
        self, backend: Backend, app_id: str = "reachy_mini_daemon", spawn: bool = True
    ):
        rr.init(app_id, spawn=spawn)
        self.app_id = app_id
        self.backend = backend

        self.recording = rr.get_global_data_recording()

        script_dir = os.path.dirname(os.path.abspath(__file__))

        urdf_path = os.path.join(script_dir, "descriptions/reachy_mini/urdf/robot.urdf")
        asset_path = os.path.join(script_dir, "descriptions/reachy_mini/urdf")

        fixed_urdf = self.set_absolute_path_to_urdf(urdf_path, asset_path)

        self.urdf_logger = URDFLogger(fixed_urdf, "ReachyMini")
        self.urdf_logger.log(recording=self.recording)

        self.running = Event()
        self.thread_log_mouvement = Thread(target=self.log_movement, daemon=True)

    def set_absolute_path_to_urdf(self, urdf_path: str, abs_path: str):
        """
        Set the absolute path to the URDF file.
        """
        with open(urdf_path, "r") as f:
            urdf_content = f.read()
        urdf_content_mod = urdf_content.replace("package://", f"file://{abs_path}/")

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".urdf") as tmp_file:
            tmp_file.write(urdf_content_mod)
            return tmp_file.name

    def start(self):
        self.thread_log_mouvement.start()

    def stop(self):
        self.running.set()

    def _get_joints(self, joint_name: str) -> urdf.Joint:
        for j in self.urdf_logger.urdf.joints:
            if j.name == joint_name:
                return j
        raise RuntimeError("Invalid joint name")

    def log_movement(self):
        """
        Log the movement data to Rerun.
        """
        antenna_left = self._get_joints("left_antenna")
        antenna_left_joint = self.urdf_logger.joint_entity_path(antenna_left)
        antenna_right = self._get_joints("right_antenna")
        antenna_right_joint = self.urdf_logger.joint_entity_path(antenna_right)

        while not self.running.is_set():
            antennas = self.backend.get_antenna_joint_positions()
            antenna_left.origin.rotation = [
                -0.0581863,
                -0.527253,
                -0.0579647 + antennas[0],
            ]

            self.urdf_logger.log_joint(
                antenna_left_joint, joint=antenna_left, recording=self.recording
            )
            antenna_right.origin.rotation = [1.5708, -1.40009 - antennas[1], -1.48353]
            self.urdf_logger.log_joint(
                antenna_right_joint, joint=antenna_right, recording=self.recording
            )
            time.sleep(0.1)
        print("Rerun logging stopped.")
