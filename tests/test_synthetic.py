import numpy as np

from edgeavi.ml.synthetic import render_sample


def test_masks_match_labels():
    rng = np.random.default_rng(0)
    for _ in range(20):
        img, m = render_sample(rng, True)
        assert img.shape == (480, 640, 3) and m.shape == (480, 640) and (m > 0).sum() > 150
        _, m0 = render_sample(rng, False)
        assert (m0 > 0).sum() == 0
