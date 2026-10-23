"""Relative contrast and local intrinsic dimension of SIFT and GloVe test queries,
measured exactly as for the Part 4 synthetic data, so the real datasets can be
placed on the Part 4 difficulty axis."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3 import config, data  # noqa: E402
from a3.bench import ground_truth, log  # noqa: E402
from a3.synth import lid_mle, relative_contrast  # noqa: E402


def main():
    out = {}
    test = slice(config.N_QUERIES_SPLIT, None)
    for name in ["sift", "glove"]:
        ds = data.load(name, config.N_BASE)
        xb, xq = ds["base"], ds["queries"]
        D, _ = ground_truth(name, xb, xq)
        rc = relative_contrast(xb, xq[test], D[test, 0])
        lid = lid_mle(D[test])
        out[name] = {"relative_contrast_median": float(np.median(rc)),
                     "lid_mle_median": float(np.median(lid)), "n_base": len(xb)}
        log(name, out[name])
    with open(config.RAW_DIR / "dataset_difficulty.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
