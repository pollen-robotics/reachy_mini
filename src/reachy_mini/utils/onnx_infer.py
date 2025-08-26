import onnxruntime  # noqa: D100


class OnnxInfer:
    """Infer an onnx model."""

    def __init__(self, onnx_model_path):
        """Initialize."""
        self.onnx_model_path = onnx_model_path
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, providers=["CPUExecutionProvider"]
        )

    def infer(self, input):
        """Run inference on the input."""
        outputs = self.ort_session.run(None, {"input": [input]})
        return outputs[0][0]
