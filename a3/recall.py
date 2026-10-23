"""Recall@k that does not punish ties at the k-th neighbour.

A returned id counts as a hit if its *true* distance to the query is no larger
than the true distance of the k-th exact neighbour (plus a tiny tolerance for
float rounding). So if two base vectors are tied at rank 10, returning either
of them is correct. Distances are recomputed in float64 from the raw vectors,
never taken from the index (IVF-PQ returns approximate distances).
"""
import numpy as np

REL_TOL = 1e-6
ABS_TOL = 1e-9


def _sq_dists(base, queries, ids):
    """Squared L2 distance from each query to each of its ids, in float64.

    ids may contain -1 (FAISS pads with -1 when it finds fewer than k results);
    those get +inf.
    """
    ids = np.asarray(ids)
    valid = ids >= 0
    safe = np.where(valid, ids, 0)
    v = base[safe].astype(np.float64)                      # (nq, k, d)
    q = np.asarray(queries, dtype=np.float64)[:, None, :]  # (nq, 1, d)
    d = ((v - q) ** 2).sum(axis=2)
    d[~valid] = np.inf
    return d


def kth_true_distance(base, queries, gt_ids, k):
    """Squared distance of the k-th exact neighbour, recomputed in float64."""
    d = _sq_dists(base, queries, np.asarray(gt_ids)[:, :k])
    return d.max(axis=1)


def recall_per_query(pred_ids, base, queries, gt_ids, k=10, kth_dist=None):
    """Fraction of the first k returned ids that are true top-k (ties allowed).

    pred_ids : (nq, >=k) ids returned by the index, -1 for missing.
    gt_ids   : (nq, >=k) exact neighbour ids, used only to get the k-th distance.
    kth_dist : optional precomputed kth_true_distance (saves time in sweeps).
    """
    pred = np.asarray(pred_ids)[:, :k]
    if kth_dist is None:
        kth_dist = kth_true_distance(base, queries, gt_ids, k)
    thr = kth_dist * (1 + REL_TOL) + ABS_TOL
    d = _sq_dists(base, queries, pred)
    hit = d <= thr[:, None]
    # a duplicated id must only count once
    srt = np.sort(pred, axis=1)
    dup_sorted = np.zeros_like(srt, dtype=bool)
    dup_sorted[:, 1:] = srt[:, 1:] == srt[:, :-1]
    order = np.argsort(pred, axis=1, kind="stable")
    dup = np.zeros_like(dup_sorted)
    np.put_along_axis(dup, order, dup_sorted, axis=1)
    hit &= ~dup
    return np.minimum(hit.sum(axis=1), k) / k


def recall_at_k(pred_ids, base, queries, gt_ids, k=10, kth_dist=None):
    """Mean recall@k over queries."""
    return float(recall_per_query(pred_ids, base, queries, gt_ids, k, kth_dist).mean())
