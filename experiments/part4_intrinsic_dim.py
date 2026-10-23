"""Part 4: own data. Vary the intrinsic dimension d_int of Gaussian data in a
fixed 128-d ambient space and see what it does to HNSW and IVF-Flat.

Settings: d_int in {8, 16, 32, 64, 128}; control = d_int 16 regenerated with a
different seed (nothing changed except the random draw). Everything else is
fixed: N, ambient dim, noise, scale, index build settings, sweeps, timing.
"""
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3 import config  # noqa: E402
from a3.bench import Queries, Recorder, exact_ms, ground_truth, log, sweep  # noqa: E402
from a3.indexes import HNSW_EF, NPROBE, build_hnsw, build_ivfflat, nlist_values  # noqa: E402
from a3.synth import lid_mle, relative_contrast, subspace_gaussian  # noqa: E402

N_QUERY = 10_000
SETTINGS = [  # (label, d_int, seed)
    ("d8", 8, config.SEED + 8),
    ("d16", 16, config.SEED + 16),
    ("d32", 32, config.SEED + 32),
    ("d64", 64, config.SEED + 64),
    ("d128", 128, config.SEED + 128),
    ("d16-control", 16, config.SEED + 1016),  # same property, new seed
]
HNSW_M = 16


def main():
    rec = Recorder("part4")
    n = config.N_SYNTH
    nlist = nlist_values(n)[0]
    stats = {}
    for label, d_int, seed in SETTINGS:
        log(f"=== {label}: d_int={d_int}, seed={seed}, n={n}")
        xb, xq = subspace_gaussian(n, N_QUERY, d_int, seed)
        gt_D, gt_I = ground_truth(f"synth_{label}_s{seed}", xb, xq)
        q = Queries(xb, xq, gt_I)
        test = slice(config.N_QUERIES_SPLIT, None)
        rc = relative_contrast(xb, xq[test], gt_D[test, 0])
        lid = lid_mle(gt_D[test])
        cap = exact_ms(xb, q)
        stats[label] = {"d_int": d_int, "seed": seed,
                        "relative_contrast_median": float(np.median(rc)),
                        "lid_mle_median": float(np.median(lid)),
                        "exact_ms_per_query": cap}
        log(f"relative contrast {np.median(rc):.3f}, LID(MLE,k=10) {np.median(lid):.1f}, "
            f"exact {cap:.2f} ms")
        extra = {"setting": label, "d_int": d_int, "seed": seed,
                 "relative_contrast_median": float(np.median(rc)),
                 "lid_mle_median": float(np.median(lid))}
        for built, values in [(build_hnsw(xb, HNSW_M), HNSW_EF),
                              (build_ivfflat(xb, nlist), [p for p in NPROBE if p <= nlist])]:
            log(f"built {built.kind} {built.params} in {built.build_time:.1f}s")
            sweep(rec, "part4", f"synth-{label}", xb, q, built, values, max_ms=cap, extra=extra)
            del built
            gc.collect()
    with open(config.RAW_DIR / "part4_summary.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
