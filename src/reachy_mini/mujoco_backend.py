import json
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from reachy_mini.io import Backend
from reachy_mini.mujoco_utils import (
    get_actuator_names,
    get_joint_addr_from_name,
    get_joint_id_from_name,
)

from .reachy_mini import SLEEP_ANTENNAS_JOINT_POSITIONS, SLEEP_HEAD_JOINT_POSITIONS

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


class MujocoBackend(Backend):
    def __init__(self, scene="empty"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(
            f"{ROOT_PATH}/descriptions/reachy_mini/mjcf/scenes/{scene}.xml"
        )
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # s, simulation timestep, 500hz
        self.decimation = 10  # -> 50hz control loop

        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_camera"
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
        step = 1
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
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
            with viewer.lock():
                self.data.qpos[self.joint_qpos_addr] = np.array(
                    SLEEP_HEAD_JOINT_POSITIONS + SLEEP_ANTENNAS_JOINT_POSITIONS
                ).reshape(-1, 1)
                self.data.ctrl[:] = np.array(
                    SLEEP_HEAD_JOINT_POSITIONS + SLEEP_ANTENNAS_JOINT_POSITIONS
                )

                # recompute all kinematics, collisions, etc.
                mujoco.mj_forward(self.model, self.data)

            # one more frame so the viewer shows your startup pose
            mujoco.mj_step(self.model, self.data)
            viewer.sync()

            # 3) now enter your normal loop
            while not self.should_stop.is_set():
                start_t = time.time()

                if step % self.decimation == 0:
                    if self.head_joint_positions is not None:
                        self.data.ctrl[:7] = self.head_joint_positions
                    if self.antenna_joint_positions is not None:
                        self.data.ctrl[-2:] = self.antenna_joint_positions

                    if self.joint_positions_publisher is not None:
                        self.joint_positions_publisher.put(
                            json.dumps(
                                {
                                    "head_joint_positions": self.get_head_joint_positions(),
                                    "antennas_joint_positions": self.get_antenna_joint_positions(),
                                }
                            ).encode("utf-8")
                        )

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                took = time.time() - start_t
                time.sleep(max(0, self.model.opt.timestep - took))
                step += 1

    def get_head_joint_positions(self):
        return self.data.qpos[self.joint_qpos_addr[:7]].flatten().tolist()

    def get_antenna_joint_positions(self):
        return self.data.qpos[self.joint_qpos_addr[-2:]].flatten().tolist()

    def set_torque(self, enabled: bool) -> None:
        pass
        pass
