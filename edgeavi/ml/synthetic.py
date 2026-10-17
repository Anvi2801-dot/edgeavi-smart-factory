"""Synthetic training-data generator that mimics the top-down Webots inspection camera.

Each sample = 640x480 BGR image + binary defect mask. Variation: lighting gain and
gradient, product colour, position/yaw jitter, noise, blur, 0-3 defects (blobs and
scratches). Keep the look consistent with webots/controllers/edgeavi_line (grey belt,
pastel box, dark patches) - then fine-tune on real Webots frames (see README).

    python -m edgeavi.ml.synthetic --out data/synthetic --train 1500 --val 300
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 480
PRODUCT_PX = 147  # 0.12 m product at ~1228 px/m


def _rot(pts: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return pts @ np.array([[c, -s], [s, c]]).T


def _background(rng) -> np.ndarray:
    img = np.full((H, W, 3), rng.uniform(35, 55), np.float32)            # floor
    belt = rng.uniform(80, 105)
    img[:, 45:595] = belt
    for y in range(int(rng.integers(0, 40)), H, 40):                    # roller lines
        img[y:y + 2, 45:595] *= 0.85
    img[:, 45:52] *= 0.7
    img[:, 588:595] *= 0.7
    return img


def render_sample(rng: np.random.Generator, defective: bool) -> tuple[np.ndarray, np.ndarray]:
    img = _background(rng)
    mask = np.zeros((H, W), np.uint8)

    cx = 320 + rng.uniform(-60, 60)
    cy = 240 + rng.uniform(-20, 20)
    yaw = np.deg2rad(rng.uniform(-12, 12))
    half = PRODUCT_PX / 2 * rng.uniform(0.93, 1.07)
    corners = _rot(np.array([[-half, -half], [half, -half], [half, half], [-half, half]]), yaw) + [cx, cy]

    # soft shadow, then product
    shadow = np.zeros((H, W), np.float32)
    cv2.fillConvexPoly(shadow, (corners + [6, 8]).astype(np.int32), 1.0)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 5)
    img *= (1 - 0.35 * shadow)[..., None]
    hsv = np.uint8([[[rng.uniform(0, 179), rng.uniform(50, 150), rng.uniform(170, 235)]]])
    colour = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    prod = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(prod, corners.astype(np.int32), 255)
    img[prod > 0] = colour

    if defective:
        for _ in range(int(rng.integers(1, 4))):
            local = rng.uniform(-half + 22, half - 22, size=2)
            centre = tuple(float(v) for v in (_rot(local[None], yaw)[0] + [cx, cy]))
            dark = tuple(float(v) for v in rng.uniform(15, 45, 3))
            ang = float(np.rad2deg(rng.uniform(0, np.pi)))
            if rng.random() < 0.5:   # blob
                axes = (int(rng.integers(8, 19)), int(rng.integers(8, 19)))
                cv2.ellipse(img, (int(centre[0]), int(centre[1])), axes, ang, 0, 360, dark, -1, cv2.LINE_AA)
                cv2.ellipse(mask, (int(centre[0]), int(centre[1])), axes, ang, 0, 360, 255, -1)
            else:                    # scratch
                size = (float(rng.uniform(40, 70)), float(rng.uniform(5, 9)))
                box = cv2.boxPoints((centre, size, ang)).astype(np.int32)
                cv2.fillConvexPoly(img, box, dark, cv2.LINE_AA)
                cv2.fillConvexPoly(mask, box, 255)
        mask &= prod  # defects only on the product

    # lighting: global gain + gradient, then sensor noise / blur
    gain = rng.uniform(0.55, 1.45)
    gx, gy = rng.uniform(-0.15, 0.15, 2)
    xs = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    ys = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    img *= gain * (1 + gx * xs + gy * ys)[..., None]
    img += rng.normal(0, rng.uniform(2, 7), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, mask


def generate(out: Path, n_train: int, n_val: int, seed: int, defect_ratio: float = 0.5) -> None:
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)
        with open(out / split / "labels.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["name", "defective", "mask_area_px"])
            for i in range(n):
                defective = bool(rng.random() < defect_ratio)
                img, mask = render_sample(rng, defective)
                name = f"{split}_{i:05d}.png"
                cv2.imwrite(str(out / split / "images" / name), img)
                cv2.imwrite(str(out / split / "masks" / name), mask)
                wr.writerow([name, int(defective), int((mask > 0).sum())])
        print(f"[synthetic] wrote {n} {split} samples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(Path(a.out), a.train, a.val, a.seed)
