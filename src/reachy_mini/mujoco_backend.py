from reachy_mini.io import Backend
import mujoco
import os
from pathlib import Path
import mujoco.viewer
import time
import json

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


def get_joint_id_from_name(model, name: str) -> int:
    """Return the id of a specified joint"""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def get_joint_addr_from_name(model, name: str) -> int:
    """Return the address of a specified joint"""
    return model.joint(name).qposadr


def get_actuator_names(model):
    actuator_names = [model.actuator(k).name for k in range(0, model.nu)]
    return actuator_names


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
        step = 0
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while True:
                start_t = time.time()

                # im = self.get_camera()
                # self.streamer_udp.send_frame(im)

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
