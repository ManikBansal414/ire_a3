"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ on SIFT and GloVe.

Build settings
  HNSW     : M in {8, 16, 32}, efConstruction 200
  IVF-Flat : nlist in {r, 4r}, r = sqrt(N) rounded to a power of two
  IVF-PQ   : same nlist values, m = d/4 sub-quantisers x 8 bits
             (32 bytes/code for SIFT, 25 for GloVe: 16x smaller than float32)
For each build the search parameter (efSearch / nprobe) is swept upward until
tune recall >= 0.999, recall stops improving, or a query costs more than an
exact search.
"""
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3 import config, data  # noqa: E402
from a3.bench import Queries, Recorder, exact_ms, ground_truth, log, sweep  # noqa: E402
from a3.indexes import (HNSW_EF, NPROBE, build_hnsw, build_ivfflat,  # noqa: E402
                        build_ivfpq, nlist_values)


def main():
    rec = Recorder("part2")
    for name in ["sift", "glove"]:
        log(f"=== {name}")
        ds = data.load(name, config.N_BASE)
        xb, xq = ds["base"], ds["queries"]
        _, gt_I = ground_truth(name, xb, xq)
        q = Queries(xb, xq, gt_I)
        cap = exact_ms(xb, q)
        log(f"exact search {cap:.2f} ms/query: sweeps stop beyond this")
        d = xb.shape[1]

        builders = []
        for M in [8, 16, 32]:
            builders.append(("HNSW", lambda M=M: build_hnsw(xb, M), HNSW_EF))
        for nl in nlist_values(len(xb)):
            builders.append(("IVF-Flat", lambda nl=nl: build_ivfflat(xb, nl),
                             [p for p in NPROBE if p <= nl]))
        for nl in nlist_values(len(xb)):
            builders.append(("IVF-PQ", lambda nl=nl: build_ivfpq(xb, nl, d // 4),
                             [p for p in NPROBE if p <= nl]))

        for kind, make, values in builders:
            built = make()
            log(f"built {kind} {built.params} in {built.build_time:.1f}s "
                f"({built.nbytes() / len(xb):.1f} bytes/vector)")
            sweep(rec, "part2", name, xb, q, built, values, max_ms=cap)
            del built
            gc.collect()


if __name__ == "__main__":
    main()
