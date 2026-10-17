"""Post-training static INT8 quantization (QDQ) with a calibration set that covers
the lighting/shadow variation of the factory (REQ: INT8 PTQ).
    python -m edgeavi.ml.quantize --samples 200
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from edgeavi.config import load_config
from edgeavi.engine.preprocess import preprocess


class FolderCalibrationReader(CalibrationDataReader):
    def __init__(self, image_dir: Path, h: int, w: int, n: int, input_name: str = "input"):
        files = sorted(image_dir.glob("*.png"))
        rng = np.random.default_rng(0)
        files = [files[i] for i in rng.permutation(len(files))[:n]]
        self._it = iter(files)
        self.h, self.w, self.name = h, w, input_name

    def get_next(self):
        f = next(self._it, None)
        if f is None:
            return None
        return {self.name: preprocess(cv2.imread(str(f)), self.h, self.w)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--samples", type=int, default=200)
    a = ap.parse_args()
    cfg = load_config(a.config)
    h, w = cfg["model"]["input_height"], cfg["model"]["input_width"]
    src, dst = cfg["model"]["fp32_onnx"], cfg["model"]["int8_onnx"]

    pre = src
    try:   # shape inference + graph clean-up recommended before static quantization
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = str(Path(tempfile.mkdtemp()) / "pre.onnx")
        quant_pre_process(src, pre, skip_symbolic_shape=True)
    except Exception as exc:
        print("[quantize] pre-process skipped:", exc)
        pre = src

    reader = FolderCalibrationReader(Path(cfg["paths"]["synthetic_dir"]) / "train" / "images", h, w, a.samples)
    quantize_static(pre, dst, reader, quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    print("INT8 model ->", dst)


if __name__ == "__main__":
    main()
