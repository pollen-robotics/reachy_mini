"""Mujoco Backend for Reachy Mini.

This module provides the MujocoBackend class for simulating the Reachy Mini robot using the MuJoCo physics engine.

It includes methods for running the simulation, getting joint positions, and controlling the robot's joints.

"""

import json
import time
from dataclasses import dataclass
from importlib.resources import files
from threading import Thread
from typing import Annotated, Any

import log_throttling
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt

import reachy_mini

from ..abstract import Backend, MotorControlMode
from .utils import (
    get_actuator_names,
    get_joint_addr_from_name,
    get_joint_id_from_name,
)
from .video_udp import UDPJPEGFrameSender

CAMERA_REACHY = "eye_camera"
CAMERA_STUDIO_CLOSE = "studio_close"
CAMERA_SIZES = {CAMERA_REACHY: (1280, 720), CAMERA_STUDIO_CLOSE: (640, 640)}


class MujocoBackend(Backend):
    """Simulated Reachy Mini using MuJoCo."""

    def __init__(
        self,
        scene: str = "empty",
        check_collision: bool = False,
        kinematics_engine: str = "AnalyticalKinematics",
        headless: bool = False,
        use_audio: bool = False,
    ) -> None:
        """Initialize the MujocoBackend with a specified scene.

        Args:
            scene (str): The name of the scene to load. Default is "empty".
            check_collision (bool): If True, enable collision checking. Default is False.
            kinematics_engine (str): Kinematics engine to use. Defaults to "AnalyticalKinematics".
            headless (bool): If True, run Mujoco in headless mode (no GUI). Default is False.
            use_audio (bool): If True, use audio. Default is False.

        """
        super().__init__(
            check_collision=check_collision,
            kinematics_engine=kinematics_engine,
            use_audio=use_audio,
        )

        self.headless = headless

        from reachy_mini.reachy_mini import (
            SLEEP_ANTENNAS_JOINT_POSITIONS,
            SLEEP_HEAD_JOINT_POSITIONS,
        )

        # Real robot convention for the order of the antennas joints is [right, left], but in mujoco it's [left, right]
        self._SLEEP_ANTENNAS_JOINT_POSITIONS = [
            SLEEP_ANTENNAS_JOINT_POSITIONS[1],
            SLEEP_ANTENNAS_JOINT_POSITIONS[0],
        ]
        self._SLEEP_HEAD_JOINT_POSITIONS = SLEEP_HEAD_JOINT_POSITIONS

        mjcf_root_path = str(
            files(reachy_mini).joinpath("descriptions/reachy_mini/mjcf/")
        )
        self.model = mujoco.MjModel.from_xml_path(
            f"{mjcf_root_path}/scenes/{scene}.xml"
        )
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # s, simulation timestep, 500hz
        self.decimation = 10  # -> 50hz control loop
        self.rendering_timestep = 0.04  # s, rendering loop # 25Hz
        self.streaming_timestep = 0.04  # s, streaming loop # 25Hz

        self.head_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "head",
        )

        self.current_head_pose = np.eye(4)

        self.joint_names = get_actuator_names(self.model)

        self.joint_ids = [
            get_joint_id_from_name(self.model, n) for n in self.joint_names
        ]
        self.joint_qpos_addr = [
            get_joint_addr_from_name(self.model, n) for n in self.joint_names
        ]

        # Disable collisions at the beginning for smoother initialization
        self.col_inds = []
        for i, type in enumerate(self.model.geom_contype):
            if type != 0:
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                # monkey-patch: the geoms in the minimal scene are named (duck_geom, table_top_collision ... ).
                # We don't disable the collision for them so that the objects don't fall to the ground at initialization
                if geom_name is None:
                    self.col_inds.append(i)
                    self.model.geom_contype[i] = 0
                    self.model.geom_conaffinity[i] = 0

    def _get_camera_id(self, camera_name: str) -> Any:
        """Get the id of the virtual camera."""
        return mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )

    def _get_renderer(self, camera_name: str) -> mujoco.Renderer:
        """Get the renderer for the virtual camera."""
        camera_size = CAMERA_SIZES[camera_name]
        return mujoco.Renderer(self.model, height=camera_size[1], width=camera_size[0])

    def rendering_loop(self, camera_name: str, port: int) -> None:
        """Offline Rendering loop for the Mujoco simulation.

        Capture the image from the virtual camera_name and send it over UDP to the port or over WebSocket to the ws_uri.
        """
        streamer = UDPJPEGFrameSender(dest_port=port)
        offscreen_renderer = self._get_renderer(camera_name)
        camera_id = self._get_camera_id(camera_name)

        while not self.should_stop.is_set():
            start_t = time.time()
            offscreen_renderer.update_scene(self.data, camera_id)

            im = offscreen_renderer.render()
            streamer.send_frame(im)

            took = time.time() - start_t
            time.sleep(max(0, self.rendering_timestep - took))

    def run(self) -> None:
        """Run the Mujoco simulation with a viewer.

        This method initializes the viewer and enters the main simulation loop.
        It updates the joint positions at a rate and publishes the joint positions.
        """
        step = 1

        if not self.headless:
            viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.distance = 0.8  # ≃ ||pos - lookat||
                viewer.cam.azimuth = 160  # degrees
                viewer.cam.elevation = -20  # degrees
                viewer.cam.lookat[:] = [0, 0, 0.15]

                # force one render with your new camera
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # im = self.get_camera()
                # self.streamer_udp.send_frame(im)

        self.data.qpos[self.joint_qpos_addr] = np.array(
            self._SLEEP_HEAD_JOINT_POSITIONS + self._SLEEP_ANTENNAS_JOINT_POSITIONS
        ).reshape(-1, 1)
        self.data.ctrl[:] = np.array(
            self._SLEEP_HEAD_JOINT_POSITIONS + self._SLEEP_ANTENNAS_JOINT_POSITIONS
        )

        # recompute all kinematics, collisions, etc.
        mujoco.mj_forward(self.model, self.data)

        for i in range(100):
            mujoco.mj_step(self.model, self.data)

        # enable collisions
        for i in self.col_inds:
            self.model.geom_contype[i] = 1
            self.model.geom_conaffinity[i] = 1

        for i in range(100):
            mujoco.mj_step(self.model, self.data)

        # one more frame so the viewer shows your startup pose
        mujoco.mj_step(self.model, self.data)
        if not self.headless:
            viewer.sync()

            rendering_thread = Thread(
                target=self.rendering_loop, args=(CAMERA_REACHY, 5005), daemon=True
            )
            rendering_thread.start()

        # Update the internal states of the IK and FK to the current configuration
        # This is important to avoid jumps when starting the robot (beore wake-up)
        self.head_kinematics.ik(self.get_mj_present_head_pose(), no_iterations=20)
        self.head_kinematics.fk(
            self.get_present_head_joint_positions(), no_iterations=20
        )

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
                # Update the Placo kinematics model to recompute passive joints
                self.update_head_kinematics_model(
                    self.current_head_joint_positions,
                    self.current_antenna_joint_positions,
                )
                self.current_head_pose = self.get_mj_present_head_pose()

                # Update the target head joint positions from IK if necessary
                # - does nothing if the targets did not change
                if self.ik_required:
                    try:
                        self.update_target_head_joints_from_ik(
                            self.target_head_pose, self.target_body_yaw
                        )
                    except ValueError as e:
                        log_throttling.by_time(self.logger, interval=0.5).warning(
                            f"IK error: {e}"
                        )

                if self.target_head_joint_positions is not None:
                    self.data.ctrl[:7] = self.target_head_joint_positions
                if self.target_antenna_joint_positions is not None:
                    self.data.ctrl[-2:] = -self.target_antenna_joint_positions

                if (
                    self.joint_positions_publisher is not None
                    and self.pose_publisher is not None
                ):
                    if not self.is_shutting_down:
                        self.joint_positions_publisher.put(
                            json.dumps(
                                {
                                    "head_joint_positions": self.current_head_joint_positions.tolist(),
                                    "antennas_joint_positions": self.current_antenna_joint_positions.tolist(),
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

                if not self.headless:
                    viewer.sync()

            mujoco.mj_step(self.model, self.data)

            took = time.time() - start_t
            time.sleep(max(0, self.model.opt.timestep - took))
            # print(f"Step {step}: took {took*1e6:.1f}us")
            step += 1

        if not self.headless:
            viewer.close()
            rendering_thread.join()

    def get_mj_present_head_pose(self) -> Annotated[npt.NDArray[np.float64], (4, 4)]:
        """Get the current head pose from the Mujoco simulation.

        Returns:
            np.ndarray: The current head pose as a 4x4 transformation matrix.

        """
        mj_current_head_pose = np.eye(4)

        mj_current_head_pose[:3, :3] = self.data.site_xmat[self.head_site_id].reshape(
            3, 3
        )
        mj_current_head_pose[:3, 3] = self.data.site_xpos[self.head_site_id]
        mj_current_head_pose[2, 3] -= 0.177
        return mj_current_head_pose

    def get_status(self) -> "MujocoBackendStatus":
        """Get the status of the Mujoco backend.

        Returns:
            dict: An empty dictionary as the Mujoco backend does not have a specific status to report.

        """
        return MujocoBackendStatus(motor_control_mode=self.get_motor_control_mode())

    def get_present_head_joint_positions(
        self,
    ) -> Annotated[npt.NDArray[np.float64], (7,)]:
        """Get the current joint positions of the head."""
        pos: npt.NDArray[np.float64] = self.data.qpos[
            self.joint_qpos_addr[:7]
        ].flatten()
        return pos

    def get_present_antenna_joint_positions(
        self,
    ) -> Annotated[npt.NDArray[np.float64], (2,)]:
        """Get the current joint positions of the antennas."""
        pos: npt.NDArray[np.float64] = self.data.qpos[
            self.joint_qpos_addr[-2:]
        ].flatten()
        return -pos

    def get_motor_control_mode(self) -> MotorControlMode:
        """Get the motor control mode."""
        return MotorControlMode.Enabled

    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        """Set the motor control mode."""
        pass

    def set_motor_torque_ids(self, ids: list[str], on: bool) -> None:
        """Set the motor torque state for specific motor names."""
        pass


@dataclass
class MujocoBackendStatus:
    """Dataclass to represent the status of the Mujoco backend.

    Empty for now, as the Mujoco backend does not have a specific status to report.
    """

    motor_control_mode: MotorControlMode
    error: str | None = None
