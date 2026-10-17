"""Export the trained FP32 checkpoint to ONNX (opset 13, fixed 1x3xHxW)."""
from __future__ import annotations

import argparse

import torch

from edgeavi.config import load_config
from edgeavi.ml.model import MobileNetV3UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    h, w = cfg["model"]["input_height"], cfg["model"]["input_width"]

    model = MobileNetV3UNet(pretrained=False)
    model.load_state_dict(torch.load(cfg["model"]["checkpoint"], map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, h, w)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=13, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, cfg["model"]["fp32_onnx"], dynamo=False, **kwargs)   # torch >= 2.5
    except TypeError:
        torch.onnx.export(model, dummy, cfg["model"]["fp32_onnx"], **kwargs)
    print("exported ->", cfg["model"]["fp32_onnx"])


if __name__ == "__main__":
    main()
