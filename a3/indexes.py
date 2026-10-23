"""Builders for the three ANN index types (plus exact search) and helpers to
set their search-time parameter and measure their serialized size."""
import time

import faiss
import numpy as np

from . import config

HNSW_EF_CONSTRUCTION = 200
PQ_POINTS_PER_CENTROID = 128


class Built:
    """An index plus what we need to know about it."""

    def __init__(self, kind, index, params, build_time, train_time=0.0, ivf=None, keep=()):
        self.kind = kind            # "HNSW", "IVF-Flat", "IVF-PQ", "IVF-PQ+rerank", "Exact"
        self.index = index          # object whose .search() is benchmarked
        self.params = params        # dict of build settings
        self.build_time = build_time  # seconds, train + add, all threads
        self.train_time = train_time
        self.ivf = ivf              # the IVF layer (for nprobe), if any
        self._keep = keep           # keep sub-objects alive

    @property
    def param_name(self):
        return "efSearch" if self.kind == "HNSW" else "nprobe"

    def set_search_param(self, value):
        if self.kind == "HNSW":
            self.index.hnsw.efSearch = int(value)
        elif self.ivf is not None:
            self.ivf.nprobe = int(value)

    def nbytes(self):
        return int(faiss.serialize_index(self.index).nbytes)


def nlist_values(n):
    """r and 4r, with r = sqrt(n) rounded to a power of two."""
    r = 2 ** int(round(np.log2(np.sqrt(n))))
    return [r, 4 * r]


def train_sample(xb, nlist, seed=config.SEED, per_centroid=64):
    n = min(len(xb), per_centroid * nlist)
    if n == len(xb):
        return xb
    rng = np.random.default_rng(seed)
    return xb[np.sort(rng.choice(len(xb), n, replace=False))]


def build_exact(xb):
    t0 = time.perf_counter()
    idx = faiss.IndexFlatL2(xb.shape[1])
    idx.add(xb)
    return Built("Exact", idx, {}, time.perf_counter() - t0)


def build_hnsw(xb, M, ef_construction=HNSW_EF_CONSTRUCTION):
    t0 = time.perf_counter()
    idx = faiss.IndexHNSWFlat(xb.shape[1], M)
    idx.hnsw.efConstruction = ef_construction
    idx.add(xb)
    return Built("HNSW", idx, {"M": M, "efConstruction": ef_construction},
                 time.perf_counter() - t0)


def build_ivfflat(xb, nlist):
    d = xb.shape[1]
    t0 = time.perf_counter()
    quant = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFFlat(quant, d, nlist)
    idx.train(train_sample(xb, nlist))
    t_train = time.perf_counter() - t0
    idx.add(xb)
    return Built("IVF-Flat", idx, {"nlist": nlist}, time.perf_counter() - t0,
                 t_train, ivf=idx, keep=(quant,))


def build_ivfpq(xb, nlist, m, nbits=8, rerank=False, k_factor=10):
    """IVF-PQ with m sub-quantisers of nbits each (m*nbits/8 bytes per code).

    rerank=True wraps it in IndexRefineFlat: the IVF-PQ index proposes
    k_factor*k candidates (100 for k=10), which are re-scored with exact
    distances on the raw float vectors kept alongside.
    """
    d = xb.shape[1]
    t0 = time.perf_counter()
    quant = faiss.IndexFlatL2(d)
    ivfpq = faiss.IndexIVFPQ(quant, d, nlist, m, nbits)
    # PQ codebooks: 128 training points per code-centroid (32,768 vectors) is
    # plenty and ~4x cheaper than the FAISS default of 256.
    ivfpq.pq.cp.max_points_per_centroid = PQ_POINTS_PER_CENTROID
    ivfpq.train(train_sample(xb, nlist))
    t_train = time.perf_counter() - t0
    params = {"nlist": nlist, "m": m, "nbits": nbits}
    if rerank:
        idx = faiss.IndexRefineFlat(ivfpq)
        idx.k_factor = k_factor
        idx.add(xb)
        params["rerank"] = k_factor * config.K
        return Built("IVF-PQ+rerank", idx, params, time.perf_counter() - t0,
                     t_train, ivf=ivfpq, keep=(quant, ivfpq))
    ivfpq.add(xb)
    return Built("IVF-PQ", ivfpq, params, time.perf_counter() - t0, t_train,
                 ivf=ivfpq, keep=(quant,))


HNSW_EF = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
NPROBE = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
