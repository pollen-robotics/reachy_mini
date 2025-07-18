"""Mujoco Backend for Reachy Mini.

This module provides the MujocoBackend class for simulating the Reachy Mini robot using the MuJoCo physics engine.

It includes methods for running the simulation, getting joint positions, and controlling the robot's joints.

"""

import json
import logging
import time
from dataclasses import dataclass
from importlib.resources import files
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

import reachy_mini
from reachy_mini.daemon.backend.abstract import Backend
from reachy_mini.daemon.backend.mujoco.utils import (
    get_actuator_names,
    get_joint_addr_from_name,
    get_joint_id_from_name,
)


class MujocoBackend(Backend):
    """Simulated Reachy Mini using MuJoCo."""

    def __init__(self, scene="empty"):
        """Initialize the MujocoBackend with a specified scene.

        Args:
            scene (str): The name of the scene to load. Default is "empty".

        """
        super().__init__()

        from reachy_mini.reachy_mini import (
            SLEEP_ANTENNAS_JOINT_POSITIONS,
            SLEEP_HEAD_JOINT_POSITIONS,
        )

        self._SLEEP_ANTENNAS_JOINT_POSITIONS = SLEEP_ANTENNAS_JOINT_POSITIONS
        self._SLEEP_HEAD_JOINT_POSITIONS = SLEEP_HEAD_JOINT_POSITIONS

        mjcf_root_path = str(
            files(reachy_mini).joinpath("descriptions/reachy_mini/mjcf/")
        )
        self.model = mujoco.MjModel.from_xml_path(  # type: ignore
            f"{mjcf_root_path}/scenes/{scene}.xml"
        )
        self.data = mujoco.MjData(self.model)  # type: ignore
        self.model.opt.timestep = 0.002  # s, simulation timestep, 500hz
        self.decimation = 10  # -> 50hz control loop

        self.camera_id = mujoco.mj_name2id(  # type: ignore
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,  # type: ignore
            "eye_camera",
        )
        # self.camera_size = (1280, 720)
        # self.offscreen_renderer = mujoco.Renderer(
        #     self.model, height=self.camera_size[1], width=self.camera_size[0]
        # )

        self.joint_names = get_actuator_names(self.model)

        self.joint_ids = [
            get_joint_id_from_name(self.model, n) for n in self.joint_names
        ]
        self.joint_qpos_addr = [
            get_joint_addr_from_name(self.model, n) for n in self.joint_names
        ]

        # self.streamer_udp = UDPJPEGFrameSender()

    def run(self):
        """Run the Mujoco simulation with a viewer.

        This method initializes the viewer and enters the main simulation loop.
        It updates the joint positions at a rate and publishes the joint positions.
        """
        step = 1
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # type: ignore
                viewer.cam.distance = 0.8  # ≃ ||pos - lookat||
                viewer.cam.azimuth = 160  # degrees
                viewer.cam.elevation = -20  # degrees
                viewer.cam.lookat[:] = [0, 0, 0.15]

                # force one render with your new camera
                mujoco.mj_step(self.model, self.data)  # type: ignore
                viewer.sync()

                # im = self.get_camera()
                # self.streamer_udp.send_frame(im)
            with viewer.lock():
                self.data.qpos[self.joint_qpos_addr] = np.array(
                    self._SLEEP_HEAD_JOINT_POSITIONS
                    + self._SLEEP_ANTENNAS_JOINT_POSITIONS
                ).reshape(-1, 1)
                self.data.ctrl[:] = np.array(
                    self._SLEEP_HEAD_JOINT_POSITIONS
                    + self._SLEEP_ANTENNAS_JOINT_POSITIONS
                )

                # recompute all kinematics, collisions, etc.
                mujoco.mj_forward(self.model, self.data)  # type: ignore

            # one more frame so the viewer shows your startup pose
            mujoco.mj_step(self.model, self.data)  # type: ignore
            viewer.sync()

            # 3) now enter your normal loop
            while not self.should_stop.is_set():
                start_t = time.time()

                if step % self.decimation == 0:
                    if self.head_joint_positions is not None:
                        self.data.ctrl[:7] = self.head_joint_positions
                    if self.antenna_joint_positions is not None:
                        self.data.ctrl[-2:] = self.antenna_joint_positions

                    if (
                        self.joint_positions_publisher is not None
                        and self.pose_publisher is not None
                    ):
                        self.joint_positions_publisher.put(
                            json.dumps(
                                {
                                    "head_joint_positions": self.get_head_joint_positions(),
                                    "antennas_joint_positions": self.get_antenna_joint_positions(),
                                }
                            ).encode("utf-8")
                        )
                        self.pose_publisher.put(
                            json.dumps(
                                {
                                    "head_pose": self.get_head_pose().tolist(),
                                }
                            ).encode("utf-8")
                        )

                mujoco.mj_step(self.model, self.data)  # type: ignore
                viewer.sync()

                took = time.time() - start_t
                time.sleep(max(0, self.model.opt.timestep - took))
                step += 1
                self.ready.set()

    def get_head_joint_positions(self):
        """Get the current joint positions of the head."""
        return self.data.qpos[self.joint_qpos_addr[:7]].flatten().tolist()

    def get_antenna_joint_positions(self):
        """Get the current joint positions of the antennas."""
        return self.data.qpos[self.joint_qpos_addr[-2:]].flatten().tolist()

    def enable_motors(self) -> None:
        """Enable the motors.

        Does nothing in the Mujoco backend as it does not have a concept of enabling/disabling motors.
        """
        pass

    def disable_motors(self) -> None:
        """Disable the motors.

        Does nothing in the Mujoco backend as it does not have a concept of enabling/disabling motors.
        """
        pass

    def close(self) -> None:
        """Close the Mujoco backend."""
        # TODO Do something in mujoco here ?
        pass

    def set_head_operation_mode(self, mode: int) -> None:
        """Set mode of operation for the head.

        This does nothing in the Mujoco backend as it does not have a concept of operation modes.
        """
        pass

    def set_antennas_operation_mode(self, mode: int) -> None:
        """Set mode of operation for the antennas.

        This does nothing in the Mujoco backend as it does not have a concept of operation modes.
        """
        pass

    def get_status(self) -> "MujocoBackendStatus":
        """Get the status of the Mujoco backend.

        Returns:
            dict: An empty dictionary as the Mujoco backend does not have a specific status to report.

        """
        return MujocoBackendStatus()


@dataclass
class MujocoBackendStatus:
    """Dataclass to represent the status of the Mujoco backend.

    Empty for now, as the Mujoco backend does not have a specific status to report.
    """

    error: Optional[str] = None
