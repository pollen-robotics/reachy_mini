from utils.onnx_infer import OnnxInfer
import numpy as np
from typing import List
import time


class NNKinematics:
    def __init__(self, fk_model_path: str, ik_model_path: str):
        self.fk_infer = OnnxInfer(fk_model_path)
        self.ik_infer = OnnxInfer(ik_model_path)

    def ik(self, pose: np.ndarray):
        return self.ik_infer.infer(pose)

    def fk(self, joint_angles: List[float]):
        return self.fk_infer.infer(joint_angles)


if __name__ == "__main__":
    nn_kin = NNKinematics(
        "assets/models/fknetwork.onnx",
        "assets/models/iknetwork.onnx",
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
