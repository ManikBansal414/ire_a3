"""Timing helpers: one query at a time, single thread, median of repeats."""
import contextlib
import statistics
import time

import faiss
import numpy as np
from threadpoolctl import threadpool_limits


@contextlib.contextmanager
def single_thread():
    """Force FAISS (OpenMP) and any BLAS library to use one thread."""
    old = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        faiss.omp_set_num_threads(old)


def time_one_at_a_time(index, queries, k, repeats, warmup=50):
    """Send queries one by one to index.search; return ids and timing stats.

    Every repeat runs the full query list and stores the ids (so each repeat
    does identical work). Wall time and process CPU time are both recorded;
    cpu/wall close to 1.0 confirms that only one core was busy.
    """
    rows = [queries[i:i + 1] for i in range(len(queries))]
    for r in rows[:warmup]:
        index.search(r, k)
    ids = np.empty((len(rows), k), dtype=np.int64)
    walls, cpus = [], []
    for _ in range(repeats):
        c0 = time.process_time()
        t0 = time.perf_counter()
        for i, r in enumerate(rows):
            ids[i] = index.search(r, k)[1][0]
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
    n = len(rows)
    med = statistics.median(walls)
    return {
        "ids": ids,
        "qps": n / med,
        "ms_per_query": 1000 * med / n,
        "qps_reps": [n / w for w in walls],
        "cpu_wall_ratio": float(np.median(np.array(cpus) / np.array(walls))),
    }


def time_batch(index, queries, k, repeats):
    """All queries in a single search call. Thread count is the caller's choice."""
    index.search(queries[:100], k)  # warm up
    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        index.search(queries, k)
        walls.append(time.perf_counter() - t0)
    med = statistics.median(walls)
    return {"qps": len(queries) / med, "ms_per_query": 1000 * med / len(queries),
            "qps_reps": [len(queries) / w for w in walls]}


def python_call_overhead_us(d, k, n=20000):
    """Cost of one index.search call that does (almost) no work: a flat index
    holding one vector. This is the fixed per-query overhead of the Python/SWIG
    path, which is included in every one-at-a-time measurement."""
    idx = faiss.IndexFlatL2(d)
    idx.add(np.zeros((1, d), dtype=np.float32))
    q = np.zeros((1, d), dtype=np.float32)
    with single_thread():
        for _ in range(1000):
            idx.search(q, k)
        t0 = time.perf_counter()
        for _ in range(n):
            idx.search(q, k)
        return 1e6 * (time.perf_counter() - t0) / n
