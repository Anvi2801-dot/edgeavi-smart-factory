"""ONNX Runtime session wrapper with provider fallback (TensorRT -> CUDA -> CPU)."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort


class InferenceEngine:
    def __init__(self, model_path: str, providers: list[str] | None = None, warmup: int = 3):
        wanted = providers or ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        chosen = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape            # [1,3,H,W]
        self.height, self.width = int(shape[2]), int(shape[3])
        print(f"[engine] model={model_path}\n[engine] active providers={self.session.get_providers()}")
        dummy = np.zeros((1, 3, self.height, self.width), np.float32)
        for _ in range(warmup):                                # first runs are slow (graph init)
            self.run(dummy)

    def run(self, tensor: np.ndarray) -> np.ndarray:
        """Returns the logit map [H, W] for a [1,3,H,W] input."""
        return self.session.run(None, {self.input_name: tensor})[0][0, 0]
