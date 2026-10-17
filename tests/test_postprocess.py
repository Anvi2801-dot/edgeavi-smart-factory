import numpy as np

from edgeavi.engine.postprocess import analyze
from edgeavi.engine.preprocess import preprocess


def test_preprocess_shape_and_range():
    x = preprocess(np.full((480, 640, 3), 255, np.uint8), 256, 320)
    assert x.shape == (1, 3, 256, 320) and x.dtype == np.float32
    assert abs(x[0, 0, 0, 0] - (1 - 0.485) / 0.229) < 1e-4


def test_threshold_logic_and_area_scaling():
    logits = np.full((256, 320), -10.0, np.float32)
    logits[100:120, 100:120] = 10.0                       # 400 model px
    scale = (480 * 640) / (256 * 320)                     # 3.75
    d = analyze(logits, (480, 640), 0.5, 150)
    assert d.status == "Defective" and abs(d.defect_area_px - round(400 * scale)) <= 2
    assert 0.9 < d.confidence <= 1.0
    assert analyze(logits, (480, 640), 0.5, 5000).status == "Good"       # configurable threshold
    g = analyze(np.full((256, 320), -10.0, np.float32), (480, 640), 0.5, 150)
    assert g.status == "Good" and g.defect_area_px == 0 and 0.9 < g.confidence <= 1.0
