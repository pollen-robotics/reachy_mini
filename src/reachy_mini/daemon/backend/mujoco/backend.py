"""Mujoco Backend for Reachy Mini.

This module provides the MujocoBackend class for simulating the Reachy Mini robot using the MuJoCo physics engine.

It includes methods for running the simulation, getting joint positions, and controlling the robot's joints.

"""

import json
import time
from dataclasses import dataclass
from importlib.resources import files
from threading import Thread
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

import reachy_mini

from ..abstract import Backend, MotorControlMode
from .utils import (
    get_actuator_names,
    get_joint_addr_from_name,
    get_joint_id_from_name,
)
from .video_udp import UDPJPEGFrameSender


class MujocoBackend(Backend):
    """Simulated Reachy Mini using MuJoCo."""

    def __init__(
        self,
        scene="empty",
        check_collision: bool = False,
        kinematics_engine: str = "Placo",
    ):
        """Initialize the MujocoBackend with a specified scene.

        Args:
            scene (str): The name of the scene to load. Default is "empty".
            check_collision (bool): If True, enable collision checking. Default is False.
            kinematics_engine (str): Kinematics engine to use. Defaults to "Placo".

        """
        super().__init__(
            check_collision=check_collision, kinematics_engine=kinematics_engine
        )

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
        self.rendering_timestep = 0.04  # s, rendering loop # 25Hz

        self.camera_id = mujoco.mj_name2id(  # type: ignore
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,  # type: ignore
            "eye_camera",
        )

        self.head_id = mujoco.mj_name2id(  # type: ignore
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,  # type: ignore
            "pp01063_stewart_plateform",
        )

        self.platform_to_head_transform = np.array(
            [
                [8.66025292e-01, 5.00000194e-01, -1.83660327e-06, -1.34282000e-02],
                [5.55111512e-16, -3.67320510e-06, -1.00000000e00, -1.20000000e-03],
                [-5.00000194e-01, 8.66025292e-01, -3.18108852e-06, 3.65883000e-02],
                [0, 0, 0, 1.00000000e00],
            ]
        )
        # remove_z_offset  = np.eye(4)
        # remove_z_offset[2, 3] = -0.177
        # self.platform_to_head_transform = self.platform_to_head_transform @ remove_z_offset

        self.current_head_pose = np.eye(4)

        # print("Joints in the model:")
        # for i in range(self.model.njoint):
        #     name = mujoco.mj_id2joint(self.model, i)
        #     print(f"  {i}: {name}")

        self.joint_names = get_actuator_names(self.model)

        self.joint_ids = [
            get_joint_id_from_name(self.model, n) for n in self.joint_names
        ]
        self.joint_qpos_addr = [
            get_joint_addr_from_name(self.model, n) for n in self.joint_names
        ]

    def rendering_loop(self):
        """Offline Rendering loop for the Mujoco simulation.

        Capture the image from the virtual Reachy's camera and send it over UDP.
        """
        streamer_udp = UDPJPEGFrameSender()
        camera_size = (1280, 720)
        offscreen_renderer = mujoco.Renderer(
            self.model, height=camera_size[1], width=camera_size[0]
        )
        while not self.should_stop.is_set():
            start_t = time.time()
            offscreen_renderer.update_scene(self.data, self.camera_id)
            im = offscreen_renderer.render()
            streamer_udp.send_frame(im)

            took = time.time() - start_t
            time.sleep(max(0, self.rendering_timestep - took))

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

            rendering_thread = Thread(target=self.rendering_loop, daemon=True)
            rendering_thread.start()

            # 3) now enter your normal loop
            while not self.should_stop.is_set():
                start_t = time.time()

                if step % self.decimation == 0:
                    # update the current states
                    self.current_head_joint_positions = (
                        self.get_present_head_joint_positions()
                    )
                    self.current_antenna_joint_positions = (
                        self.get_present_antenna_joint_positions()
                    )
                    self.current_head_pose = self.get_mj_present_head_pose()

                    # Update the target head joint positions from IK if necessary
                    # - does nothing if the targets did not change
                    if self.ik_required:
                        # Use effective targets (absolute + relative offsets)
                        effective_pose, effective_yaw = (
                            self.get_effective_head_pose_and_yaw()
                        )
                        try:
                            self.update_target_head_joints_from_ik(
                                effective_pose, effective_yaw
                            )
                        except Exception as e:
                            print("IK error:", e)

                    if self.target_head_joint_positions is not None:
                        self.data.ctrl[:7] = self.target_head_joint_positions
                    if self.target_antenna_joint_positions is not None:
                        # Use effective antenna targets (absolute + relative offsets)
                        effective_antenna_positions = (
                            self.get_effective_antenna_positions()
                        )
                        self.data.ctrl[-2:] = effective_antenna_positions

                    if (
                        self.joint_positions_publisher is not None
                        and self.pose_publisher is not None
                    ):
                        self.joint_positions_publisher.put(
                            json.dumps(
                                {
                                    "head_joint_positions": self.current_head_joint_positions,
                                    "antennas_joint_positions": self.current_antenna_joint_positions,
                                }
                            ).encode("utf-8")
                        )
                        self.pose_publisher.put(
                            json.dumps(
                                {
                                    "head_pose": self.get_present_head_pose().tolist(),
                                }
                            ).encode("utf-8")
                        )
                        self.ready.set()

                    viewer.sync()

                mujoco.mj_step(self.model, self.data)  # type: ignore

                took = time.time() - start_t
                time.sleep(max(0, self.model.opt.timestep - took))
                # print(f"Step {step}: took {took*1000:.1f}ms")
                step += 1

    def get_mj_present_head_pose(self) -> np.ndarray:
        """Get the current head pose from the Mujoco simulation.

        Returns:
            np.ndarray: The current head pose as a 4x4 transformation matrix.

        """
        mj_current_head_pose = np.eye(4)
        mj_current_head_pose[:3, :3] = self.data.xmat[self.head_id].reshape(3, 3)
        mj_current_head_pose[:3, 3] = self.data.xpos[self.head_id]
        mj_current_head_pose = mj_current_head_pose @ self.platform_to_head_transform
        mj_current_head_pose[2, 3] -= 0.177
        return mj_current_head_pose

    def close(self) -> None:
        """Close the Mujoco backend."""
        # TODO Do something in mujoco here ?
        pass

    def get_status(self) -> "MujocoBackendStatus":
        """Get the status of the Mujoco backend.

        Returns:
            dict: An empty dictionary as the Mujoco backend does not have a specific status to report.

        """
        return MujocoBackendStatus()

    def get_present_head_joint_positions(self):
        """Get the current joint positions of the head."""
        return self.data.qpos[self.joint_qpos_addr[:7]].flatten().tolist()

    def get_present_antenna_joint_positions(self):
        """Get the current joint positions of the antennas."""
        return self.data.qpos[self.joint_qpos_addr[-2:]].flatten().tolist()

    def get_motor_control_mode(self) -> MotorControlMode:
        """Get the motor control mode."""
        return MotorControlMode.Enabled

    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        """Set the motor control mode."""
        pass


@dataclass
class MujocoBackendStatus:
    """Dataclass to represent the status of the Mujoco backend.

    Empty for now, as the Mujoco backend does not have a specific status to report.
    """

    error: Optional[str] = None
