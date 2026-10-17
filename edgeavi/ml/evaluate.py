"""Compare FP32 vs INT8 on the validation set: mask IoU, image-level accuracy, agreement.
    python -m edgeavi.ml.evaluate
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from edgeavi.config import load_config
from edgeavi.engine.inference import InferenceEngine
from edgeavi.engine.postprocess import analyze
from edgeavi.engine.preprocess import preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    thr = cfg["inference"]["defect_area_threshold_px"]
    val = Path(cfg["paths"]["synthetic_dir"]) / "val"
    names = [r["name"] for r in csv.DictReader(open(val / "labels.csv"))]
    engines = {k: InferenceEngine(cfg["model"][f"{k}_onnx"], ["CPUExecutionProvider"]) for k in ("fp32", "int8")}
    stats = {k: {"iou": [], "correct": 0, "pred": []} for k in engines}

    for n in names:
        img = cv2.imread(str(val / "images" / n))
        gt = cv2.imread(str(val / "masks" / n), 0)
        gt_def = int((gt > 0).sum()) >= thr
        for k, eng in engines.items():
            res = analyze(eng.run(preprocess(img, eng.height, eng.width)), img.shape[:2], 0.5, thr)
            pm = cv2.resize(res.mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if gt.any():
                g = gt > 0
                stats[k]["iou"].append((pm & g).sum() / max((pm | g).sum(), 1))
            stats[k]["correct"] += (res.status == "Defective") == gt_def
            stats[k]["pred"].append(res.status)
    for k, s in stats.items():
        print(f"{k.upper():5s} image-level accuracy {100 * s['correct'] / len(names):.2f}% | "
              f"mean mask IoU (defective imgs) {np.mean(s['iou']):.3f}")
    agree = np.mean([a == b for a, b in zip(stats["fp32"]["pred"], stats["int8"]["pred"])])
    print(f"FP32/INT8 decision agreement {100 * agree:.2f}%")
    size = lambda p: Path(p).stat().st_size / 1e6
    print(f"model size: FP32 {size(cfg['model']['fp32_onnx']):.1f} MB -> INT8 {size(cfg['model']['int8_onnx']):.1f} MB")


if __name__ == "__main__":
    main()
