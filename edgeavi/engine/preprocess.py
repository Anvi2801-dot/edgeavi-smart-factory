"""REQ-3: normalise pixels and convert a BGR frame into an NCHW float32 tensor."""
from __future__ import annotations

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_SCALE = (1.0 / (255.0 * STD)).astype(np.float32)
_SHIFT = (-MEAN / STD).astype(np.float32)


def preprocess(frame_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    small = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = rgb * _SCALE + _SHIFT                      # (x/255 - mean)/std in one pass
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
