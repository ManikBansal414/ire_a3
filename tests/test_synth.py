"""The Part 4 generator must be seeded (same seed -> same data) and must keep
the scale fixed while only the intrinsic dimension changes."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3.synth import lid_mle, subspace_gaussian  # noqa: E402


def test_same_seed_same_data():
    a = subspace_gaussian(1000, 10, 16, seed=7)
    b = subspace_gaussian(1000, 10, 16, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_different_seed_different_data():
    a = subspace_gaussian(1000, 10, 16, seed=7)[0]
    b = subspace_gaussian(1000, 10, 16, seed=8)[0]
    assert not np.array_equal(a, b)


def test_scale_independent_of_dint():
    for d_int in (8, 32, 128):
        xb, _ = subspace_gaussian(20000, 10, d_int, seed=1)
        assert xb.shape == (20000, 128)
        assert abs((xb.astype(np.float64) ** 2).sum(1).mean() - 1.001) < 0.03


def test_data_lies_near_subspace():
    xb, _ = subspace_gaussian(5000, 10, 8, seed=3)
    sv = np.linalg.svd(xb - xb.mean(0), compute_uv=False) ** 2
    assert sv[:8].sum() / sv.sum() > 0.99


def test_lid_estimate_tracks_dimension():
    # LID of uniform-ish Gaussian data grows with its true dimension
    rng = np.random.default_rng(0)
    est = []
    for d in (2, 8):
        x = rng.standard_normal((20000, d))
        q = rng.standard_normal((200, d))
        d2 = ((q[:, None] - x[None]) ** 2).sum(2)
        knn = np.sort(d2, 1)[:, :10]
        est.append(np.median(lid_mle(knn)))
    assert est[0] < est[1]
