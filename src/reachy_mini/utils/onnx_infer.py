import onnxruntime


class OnnxInfer:
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, providers=["CPUExecutionProvider"]
        )

    def infer(self, input):
        outputs = self.ort_session.run(None, {"input": [input]})
        return outputs[0][0]


if __name__ == "__main__":
    import numpy as np

    onnx_infer = OnnxInfer(
        "/home/antoine/Pollen/reachy_mini_nn_kinematics/fknetwork.onnx"
    )
    for i in range(100):
        input = np.random.random(7).astype(np.float32)
        print(input)

        print("input:", input)
        output = onnx_infer.infer(input)
        print("output:", output)
        print("==")
