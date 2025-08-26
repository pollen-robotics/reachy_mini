from reachy_mini.utils.onnx_infer import OnnxInfer
import numpy as np
from typing import List
import time
from scipy.spatial.transform import Rotation as R


class NNKinematics:
    def __init__(self, models_root_path: str):
        self.fk_model_path = f"{models_root_path}/fknetwork.onnx"
        self.ik_model_path = f"{models_root_path}/iknetwork.onnx"
        self.fk_infer = OnnxInfer(self.fk_model_path)
        self.ik_infer = OnnxInfer(self.ik_model_path)

        self.start_body_yaw = 0.0  # No used, kept for compatibility

    def ik(
        self,
        pose: np.ndarray,
        body_yaw: float = 0.0,
        check_collision: bool = False,
        no_iterations: int = 0,
    ):
        """body_yaw, check_collision and no_iterations are not used by NNKinematics.

        We keep them for compatibility with the other kinematics engines
        """
        x, y, z = pose[:3, 3][0], pose[:3, 3][1], pose[:3, 3][2]
        roll, pitch, yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz")

        input = [x, y, z, roll, pitch, yaw]

        return self.ik_infer.infer(input)

    def fk(
        self,
        joint_angles: List[float],
        check_collision: bool = False,
        no_iterations: int = 0,
    ):
        """check_collision and no_iterations are not used by NNKinematics.

        We keep them for compatibility with the other kinematics engines
        """

        x, y, z, roll, pitch, yaw = self.fk_infer.infer(joint_angles)
        pose = np.eye(4)
        pose[:3, 3] = [x, y, z]
        pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        return pose


if __name__ == "__main__":
    nn_kin = NNKinematics(
        "assets/models",
    )

    times_fk = []
    times_ik = []
    for i in range(1000):
        fk_input = np.random.random(7).astype(np.float32)
        ik_input = np.random.random(6).astype(np.float32)

        fk_s = time.time()
        fk_output = nn_kin.fk(fk_input)
        times_fk.append(time.time() - fk_s)

        ik_s = time.time()
        ik_output = nn_kin.ik(ik_input)
        times_ik.append(time.time() - ik_s)

    print(f"Average FK inference time: {np.mean(times_fk) * 1e6} µs")
    print(f"Average IK inference time: {np.mean(times_ik) * 1e6} µs")
