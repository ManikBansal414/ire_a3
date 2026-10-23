"""Part 3 (option: Compression): on SIFT, vary the number of PQ sub-vectors m
and compare plain IVF-PQ with IVF-PQ + exact re-ranking of the top 100.

Fixed: nlist = r (the smaller nlist of Part 2), 8 bits per sub-quantiser,
same training sample, same queries. Varied: m in {8, 16, 32, 64}, rerank on/off.
"""
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3 import config, data  # noqa: E402
from a3.bench import Queries, Recorder, exact_ms, ground_truth, log, sweep  # noqa: E402
from a3.indexes import NPROBE, build_ivfpq, nlist_values  # noqa: E402

M_VALUES = [8, 16, 32, 64]


def main():
    rec = Recorder("part3")
    ds = data.load("sift", config.N_BASE)
    xb, xq = ds["base"], ds["queries"]
    _, gt_I = ground_truth("sift", xb, xq)
    q = Queries(xb, xq, gt_I)
    cap = exact_ms(xb, q)
    nlist = nlist_values(len(xb))[0]
    log(f"SIFT n={len(xb)}, nlist={nlist}, exact {cap:.2f} ms/query")
    for m in M_VALUES:
        for rerank in (False, True):
            built = build_ivfpq(xb, nlist, m, rerank=rerank)
            log(f"built {built.kind} {built.params} in {built.build_time:.1f}s, "
                f"{built.nbytes() / len(xb):.1f} bytes/vector")
            sweep(rec, "part3", "sift", xb, q, built, [p for p in NPROBE if p <= nlist],
                  max_ms=cap, extra={"m": m, "rerank": rerank})
            del built
            gc.collect()


if __name__ == "__main__":
    main()
