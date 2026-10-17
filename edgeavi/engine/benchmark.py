"""Measure end-to-end per-frame latency (preprocess + inference + postprocess).
    python -m edgeavi.engine.benchmark --model models/model_int8.onnx --n 200
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from edgeavi.config import load_config
from edgeavi.engine.inference import InferenceEngine
from edgeavi.engine.postprocess import analyze
from edgeavi.engine.preprocess import preprocess
from edgeavi.ml.synthetic import render_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    cfg = load_config(a.config)
    eng = InferenceEngine(a.model or cfg["model"]["int8_onnx"], cfg["inference"]["providers"])
    rng = np.random.default_rng(1)
    frames = [render_sample(rng, bool(i % 2))[0] for i in range(20)]
    times = []
    for i in range(a.n):
        f = frames[i % len(frames)]
        t = time.perf_counter()
        analyze(eng.run(preprocess(f, eng.height, eng.width)), f.shape[:2], 0.5, 150)
        times.append((time.perf_counter() - t) * 1000)
    t = np.array(times)
    budget = cfg["inference"]["latency_budget_ms"]
    print(f"mean {t.mean():.1f} ms | p50 {np.percentile(t, 50):.1f} | p95 {np.percentile(t, 95):.1f} | max {t.max():.1f}")
    print(f"budget {budget} ms -> {'PASS' if np.percentile(t, 95) < budget else 'FAIL'} (p95)")


if __name__ == "__main__":
    main()
