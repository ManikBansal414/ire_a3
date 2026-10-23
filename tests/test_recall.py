"""Unit tests for the recall@k function (run with `python -m pytest tests`)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3.recall import recall_at_k, recall_per_query  # noqa: E402

K = 3


def line_data():
    """Base points on a line at 0,1,2,...,9; one query at -0.1."""
    base = np.arange(10, dtype=np.float32)[:, None]
    q = np.array([[-0.1]], dtype=np.float32)
    gt = np.array([[0, 1, 2]])
    return base, q, gt


def test_perfect():
    base, q, gt = line_data()
    assert recall_at_k(gt, base, q, gt, k=K) == 1.0


def test_order_does_not_matter():
    base, q, gt = line_data()
    assert recall_at_k(np.array([[2, 0, 1]]), base, q, gt, k=K) == 1.0


def test_all_wrong():
    base, q, gt = line_data()
    assert recall_at_k(np.array([[7, 8, 9]]), base, q, gt, k=K) == 0.0


def test_partial():
    base, q, gt = line_data()
    assert abs(recall_at_k(np.array([[0, 1, 9]]), base, q, gt, k=K) - 2 / 3) < 1e-12


def test_tie_at_kth_is_not_a_mistake():
    # points 2 and 3 are both at distance 1 from the query at 1.0 ... make a tie:
    base = np.array([[0.0], [1.0], [2.0], [2.0], [5.0]], dtype=np.float32)
    q = np.array([[0.0]], dtype=np.float32)
    gt = np.array([[0, 1, 2]])          # ground truth picked id 2, id 3 is tied with it
    pred = np.array([[0, 1, 3]])        # index returned the tied twin instead
    assert recall_at_k(pred, base, q, gt, k=K) == 1.0


def test_tie_does_not_allow_farther_points():
    base = np.array([[0.0], [1.0], [2.0], [2.0], [2.0001]], dtype=np.float32)
    q = np.array([[0.0]], dtype=np.float32)
    gt = np.array([[0, 1, 2]])
    pred = np.array([[0, 1, 4]])        # 4 is slightly farther than the 3rd neighbour
    assert abs(recall_at_k(pred, base, q, gt, k=K) - 2 / 3) < 1e-12


def test_ties_capped_at_k():
    # many tied points: returning k of them is recall 1, never above 1
    base = np.zeros((6, 2), dtype=np.float32)
    q = np.zeros((1, 2), dtype=np.float32)
    gt = np.array([[0, 1, 2]])
    assert recall_at_k(np.array([[3, 4, 5]]), base, q, gt, k=K) == 1.0


def test_missing_results_padded_with_minus_one():
    base, q, gt = line_data()
    assert abs(recall_at_k(np.array([[0, -1, -1]]), base, q, gt, k=K) - 1 / 3) < 1e-12


def test_duplicates_count_once():
    base, q, gt = line_data()
    assert abs(recall_at_k(np.array([[0, 0, 0]]), base, q, gt, k=K) - 1 / 3) < 1e-12


def test_only_first_k_returned_are_used():
    base, q, gt = line_data()
    pred = np.array([[0, 9, 8, 1, 2]])   # correct ids beyond position k are ignored
    assert abs(recall_at_k(pred, base, q, gt, k=K) - 1 / 3) < 1e-12


def test_mean_over_queries():
    base = np.arange(10, dtype=np.float32)[:, None]
    q = np.array([[-0.1], [9.1]], dtype=np.float32)
    gt = np.array([[0, 1, 2], [9, 8, 7]])
    pred = np.array([[0, 1, 2], [0, 1, 2]])
    per = recall_per_query(pred, base, q, gt, k=K)
    assert per.tolist() == [1.0, 0.0]
    assert recall_at_k(pred, base, q, gt, k=K) == 0.5


def test_float_rounding_tolerated():
    # a value that equals the kth distance up to float32 rounding is a hit
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 16)).astype(np.float32)
    q = rng.normal(size=(1, 16)).astype(np.float32)
    d = ((base - q) ** 2).sum(1)
    order = np.argsort(d)
    base = np.vstack([base, base[order[2]][None]])   # exact copy of the 3rd neighbour
    gt = order[None, :3]
    pred = np.array([[order[0], order[1], 50]])       # returns the copy
    assert recall_at_k(pred, base, q, gt, k=K) == 1.0


def test_against_bruteforce_random():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 8)).astype(np.float32)
    q = rng.normal(size=(20, 8)).astype(np.float32)
    d = ((base[None] - q[:, None]) ** 2).sum(2)
    gt = np.argsort(d, axis=1)[:, :K]
    pred = rng.integers(0, 200, size=(20, K))
    expected = []
    for i in range(20):
        kth = np.sort(d[i])[K - 1]
        expected.append(len({p for p in pred[i] if d[i, p] <= kth}) / K)
    assert abs(recall_at_k(pred, base, q, gt, k=K) - np.mean(expected)) < 1e-12
