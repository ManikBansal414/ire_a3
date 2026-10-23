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


def time_one_at_a_time(index, queries, k, repeats, warmup=50, min_seconds=0.5):
    """Send queries one by one to index.search; return ids and timing stats.

    Each repeat runs the whole query list, and keeps cycling through it until
    at least min_seconds have passed (fast indexes finish 5,000 queries in
    ~0.1 s, too short to time reliably on a laptop). QPS of a repeat = queries
    answered / wall time; the median over repeats is reported. Process CPU
    time is recorded too: cpu/wall close to 1.0 confirms one busy core.
    """
    rows = [queries[i:i + 1] for i in range(len(queries))]
    for r in rows[:warmup]:
        index.search(r, k)
    ids = np.empty((len(rows), k), dtype=np.int64)
    rates, ratios, walls = [], [], []
    for _ in range(repeats):
        done = 0
        c0 = time.process_time()
        t0 = time.perf_counter()
        while True:
            for i, r in enumerate(rows):
                ids[i] = index.search(r, k)[1][0]
            done += len(rows)
            wall = time.perf_counter() - t0
            if wall >= min_seconds:
                break
        cpu = time.process_time() - c0
        rates.append(done / wall)
        ratios.append(cpu / wall)
        walls.append(wall)
    qps = statistics.median(rates)
    return {
        "ids": ids,
        "qps": qps,
        "ms_per_query": 1000 / qps,
        "qps_reps": rates,
        "cpu_wall_ratio": float(np.median(ratios)),
        "repeat_seconds": float(np.median(walls)),
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
