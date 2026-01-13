"""Neural Network based FK/IK."""

import time
from typing import Annotated

import numpy as np
import numpy.typing as npt
import onnxruntime
from scipy.spatial.transform import Rotation as R


class NNKinematics:
    """Neural Network based FK/IK. Fitted from PlacoKinematics data."""

    def __init__(self, models_root_path: str):
        """Initialize."""
        self.fk_model_path = f"{models_root_path}/fknetwork.onnx"
        self.ik_model_path = f"{models_root_path}/iknetwork.onnx"
        self.fk_infer = OnnxInfer(self.fk_model_path)
        self.ik_infer = OnnxInfer(self.ik_model_path)

        self.automatic_body_yaw = False  # Not used, kept for compatibility

    def ik(
        self,
        pose: Annotated[npt.NDArray[np.float64], (4, 4)],
        body_yaw: float = 0.0,
        check_collision: bool = False,
        no_iterations: int = 0,
    ) -> Annotated[npt.NDArray[np.float64], (7,)]:
        """check_collision and no_iterations are not used by NNKinematics.

        We keep them for compatibility with the other kinematics engines
        """
        x, y, z = pose[:3, 3][0], pose[:3, 3][1], pose[:3, 3][2]
        roll, pitch, yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz")

        yaw += body_yaw
        input = np.array([x, y, z, roll, pitch, yaw])

        joints = self.ik_infer.infer(input)
        joints[0] += body_yaw

        return joints

    def fk(
        self,
        joint_angles: Annotated[npt.NDArray[np.float64], (7,)],
        check_collision: bool = False,
        no_iterations: int = 0,
    ) -> Annotated[npt.NDArray[np.float64], (4, 4)]:
        """check_collision and no_iterations are not used by NNKinematics.

        We keep them for compatibility with the other kinematics engines
        """
        x, y, z, roll, pitch, yaw = self.fk_infer.infer(joint_angles)
        pose = np.eye(4)
        pose[:3, 3] = [x, y, z]
        pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        return pose

    def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
        """Set the automatic body yaw.

        Args:
            automatic_body_yaw (bool): Whether to enable automatic body yaw.

        """
        self.automatic_body_yaw = automatic_body_yaw


class OnnxInfer:
    """Infer an onnx model."""

    def __init__(self, onnx_model_path: str) -> None:
        """Initialize."""
        self.onnx_model_path = onnx_model_path
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, providers=["CPUExecutionProvider"]
        )

    def infer(self, input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Run inference on the input."""
        outputs = self.ort_session.run(None, {"input": [input]})
        res: npt.NDArray[np.float64] = outputs[0][0]
        return res


if __name__ == "__main__":
    nn_kin = NNKinematics(
        "assets/models",
    )

    times_fk: list[float] = []
    times_ik: list[float] = []
    for i in range(1000):
        fk_input = np.random.random(7).astype(np.float64)
        # ik_input = np.random.random(6).astype(np.float64)

        fk_s = time.time()
        fk_output = nn_kin.fk(fk_input)
        times_fk.append(time.time() - fk_s)

        # ik_s = time.time()
        # ik_output = nn_kin.ik(ik_input)
        # times_ik.append(time.time() - ik_s)

    print(f"Average FK inference time: {np.mean(times_fk) * 1e6} µs")
    # print(f"Average IK inference time: {np.mean(times_ik) * 1e6} µs")
