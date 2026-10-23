"""Ground truth, parameter sweeps and result rows."""
import json
import platform
import time
from datetime import datetime

import faiss
import numpy as np
import pandas as pd

from . import config
from .recall import kth_true_distance, recall_at_k
from .timing import single_thread, time_one_at_a_time


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def exact_knn(xb, xq, k):
    """Exact k-NN with a FAISS flat index (all threads, batched)."""
    idx = faiss.IndexFlatL2(xb.shape[1])
    idx.add(xb)
    D, I = idx.search(xq, k)
    return D, I


def ground_truth(tag, xb, xq, k=config.K):
    """Exact neighbours for every query, cached on disk by (tag, n, nq)."""
    path = config.CACHE_DIR / f"gt_{tag}_{len(xb)}_{len(xq)}_k{k}.npz"
    if path.exists():
        z = np.load(path)
        return z["D"], z["I"]
    D, I = exact_knn(xb, xq, k)
    np.savez(path, D=D, I=I)
    return D, I


def exact_ms(xb, q):
    """Exact one-at-a-time cost on 200 test queries: the cap for sweeps."""
    from .indexes import build_exact
    e = build_exact(xb)
    with single_thread():
        t = time_one_at_a_time(e.index, q.test[:200], config.K, repeats=1, warmup=10)
    return t["ms_per_query"]


class Queries:
    """Tune / test split with ground truth and precomputed k-th distances."""

    def __init__(self, xb, xq, gt_ids, k=config.K):
        n, h = config.N_QUERIES_SPLIT, config.N_EVAL_HALF
        self.tune, self.test = xq[:n][:h], xq[n:][:h]
        self.gt_tune, self.gt_test = gt_ids[:n][:h], gt_ids[n:][:h]
        self.kth_tune = kth_true_distance(xb, self.tune, self.gt_tune, k)
        self.kth_test = kth_true_distance(xb, self.test, self.gt_test, k)
        self.time_tune = self.tune[: config.N_TIME_TUNE]
        self.time_test = self.test[: config.N_TIME_TEST]


def machine_info():
    import threadpoolctl
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "faiss": faiss.__version__,
        "numpy": np.__version__,
        "omp_max_threads": faiss.omp_get_max_threads(),
        "blas": [(i["internal_api"], i.get("version")) for i in threadpoolctl.threadpool_info()],
    }


class Recorder:
    """Appends result rows to results/raw/<name>.csv (rewritten after every row,
    so a crash keeps everything measured so far)."""

    def __init__(self, name):
        self.path = config.RAW_DIR / f"{name}.csv"
        self.rows = []

    def add(self, **row):
        row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        row["quick_mode"] = config.QUICK
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


def base_row(part, dataset, xb, built, q):
    return {
        "part": part,
        "dataset": dataset,
        "n_base": len(xb),
        "dim": xb.shape[1],
        "index": built.kind,
        "build_params": json.dumps(built.params, sort_keys=True),
        "build_time_s": round(built.build_time, 3),
        "train_time_s": round(built.train_time, 3),
        "index_bytes": built.nbytes(),
        "bytes_per_vector": built.nbytes() / len(xb),
        "n_tune_queries": len(q.tune),
        "n_test_queries": len(q.test),
        "n_timed_queries": len(q.time_test),
        "n_repeats": config.N_REPEATS,
    }


def distance_computations(built, n, k):
    """Distance computations counted by FAISS since the last stats reset
    (vector-to-query distances; IVF also scans nlist centroids per query, and
    re-ranking adds k_factor*k exact distances, which are not included)."""
    if built.kind == "HNSW":
        return faiss.cvar.hnsw_stats.ndis
    if built.ivf is not None:
        return faiss.cvar.indexIVF_stats.ndis
    return float("nan")


def evaluate_point(built, xb, q, value, k=config.K):
    """Recall on tune and test halves + single-thread one-at-a-time timing."""
    built.set_search_param(value)
    # recall on the full halves (batched, all threads: same results, just faster)
    _, I_tune = built.index.search(q.tune, k)
    faiss.cvar.hnsw_stats.reset()
    faiss.cvar.indexIVF_stats.reset()
    _, I_test = built.index.search(q.test, k)
    ndis = distance_computations(built, len(xb), k) / len(q.test)
    r_tune = recall_at_k(I_tune, xb, q.tune, q.gt_tune, k, q.kth_tune)
    r_test = recall_at_k(I_test, xb, q.test, q.gt_test, k, q.kth_test)
    with single_thread():
        t_tune = time_one_at_a_time(built.index, q.time_tune, k, repeats=1,
                                    min_seconds=config.MIN_REPEAT_S / 4)
        t_test = time_one_at_a_time(built.index, q.time_test, k, repeats=config.N_REPEATS,
                                    min_seconds=config.MIN_REPEAT_S)
    n_t = len(q.time_test)
    r_timed = recall_at_k(t_test["ids"], xb, q.time_test, q.gt_test[:n_t], k, q.kth_test[:n_t])
    return {
        "param_name": built.param_name,
        "param_value": value,
        "recall_tune": r_tune,
        "recall_test": r_test,
        "recall_timed_subset": r_timed,
        "qps_tune": t_tune["qps"],
        "qps_test": t_test["qps"],
        "ms_per_query": t_test["ms_per_query"],
        "qps_reps": ";".join(f"{x:.1f}" for x in t_test["qps_reps"]),
        "cpu_wall_ratio": round(t_test["cpu_wall_ratio"], 3),
        "repeat_seconds": round(t_test["repeat_seconds"], 3),
        "ndis_per_query": ndis,
    }


def sweep(rec, part, dataset, xb, q, built, values, max_ms=None, stop_recall=config.STOP_RECALL,
          plateau=0.005, extra=None):
    """Run evaluate_point over increasing parameter values.

    Stops early (decided on the *tune* half, never on test) when tune recall
    reaches stop_recall, when recall has stopped improving (gain < plateau over
    the last two steps, which happens for IVF-PQ), or when a query takes longer
    than max_ms (slower than exact search is pointless).
    """
    base = base_row(part, dataset, xb, built, q)
    if extra:
        base.update(extra)
    hist = []
    for v in values:
        t0 = time.perf_counter()
        res = evaluate_point(built, xb, q, v)
        rec.add(**base, **res)
        log(f"  {built.kind} {built.params} {res['param_name']}={v}: "
            f"recall tune {res['recall_tune']:.4f} test {res['recall_test']:.4f} "
            f"qps {res['qps_test']:.0f} ({time.perf_counter() - t0:.0f}s)")
        hist.append(res["recall_tune"])
        if res["recall_tune"] >= stop_recall:
            break
        if len(hist) >= 3 and hist[-1] - hist[-3] < plateau and hist[-1] < 0.99:
            break
        if max_ms is not None and res["ms_per_query"] > max_ms:
            break
