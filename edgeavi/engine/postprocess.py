"""REQ-7 / REQ-8: logits -> mask -> flaw area (original pixels) -> Good/Defective."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_KERNEL = np.ones((3, 3), np.uint8)


@dataclass
class Analysis:
    status: str               # "Good" | "Defective"
    defect_area_px: int       # in original-frame pixels
    confidence: float         # 0..1
    mask: np.ndarray          # uint8 {0,1} at model resolution


def analyze(logits: np.ndarray, frame_hw: tuple[int, int], mask_thr: float, area_thr_px: int) -> Analysis:
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    mask = (prob > mask_thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)     # drop isolated noisy pixels
    scale = (frame_hw[0] * frame_hw[1]) / float(mask.size)
    area = int(round(int(mask.sum()) * scale))
    defective = area >= area_thr_px
    if defective:
        conf = float(prob[mask > 0].mean()) if mask.any() else float(prob.max())
    else:
        conf = float(1.0 - prob.max())                          # how sure we are nothing is there
    return Analysis("Defective" if defective else "Good", area, min(max(conf, 0.0), 1.0), mask)


def render_overlay(frame_bgr: np.ndarray, mask: np.ndarray, status: str, product_id: str, latency_ms: float) -> np.ndarray:
    out = frame_bgr.copy()
    if mask.any():
        m = cv2.resize(mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[m] = (0.4 * out[m] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    colour = (0, 0, 255) if status == "Defective" else (0, 200, 0)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(out, f"{product_id}  {status}  {latency_ms:.1f} ms", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    return out
