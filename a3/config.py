"""Paths and run-mode settings shared by every experiment script.

Two modes:
  full  - the numbers in the report (1M SIFT, 1.18M GloVe, 200K synthetic).
  quick - same pipeline on a small subset (first 100K base vectors, 50K
          synthetic, fewer timing queries) so `run_all.sh --quick` finishes
          in well under 15 minutes on a laptop.

The mode is read from the A3_QUICK environment variable (set by run_all.sh).
"""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("A3_DATA", CODE_DIR.parent / "data"))

QUICK = os.environ.get("A3_QUICK", "0") == "1"

_default_results = CODE_DIR.parent / ("results_quick" if QUICK else "results")
RESULTS_DIR = Path(os.environ.get("A3_RESULTS", _default_results))
RAW_DIR = RESULTS_DIR / "raw"          # one csv per part, merged into results.csv
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
CACHE_DIR = Path(os.environ.get("A3_CACHE", CODE_DIR.parent / "cache"))  # ground truth cache

for _d in (RAW_DIR, FIG_DIR, TABLE_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

K = 10                     # recall@10 everywhere
N_QUERIES_SPLIT = 5000     # first 5000 queries = tune, last 5000 = test

if QUICK:
    N_BASE = 100_000       # first 100K base vectors
    N_EVAL_HALF = 2000     # queries used from each half (tune / test) for recall
    N_TIME_TEST = 300      # test queries used in the timed loop
    N_TIME_TUNE = 100      # tune queries used for (cheap) timing during selection
    N_REPEATS = 3
    N_SYNTH = 50_000
    STOP_RECALL = 0.99     # sweeps stop once tune recall reaches this
    MIN_REPEAT_S = 0.1     # a timed repeat cycles the queries for at least this long
else:
    N_BASE = None          # all vectors
    N_EVAL_HALF = 5000     # all 5,000 queries of each half
    N_TIME_TEST = 5000     # all test queries are timed
    N_TIME_TUNE = 1000
    N_REPEATS = 3
    N_SYNTH = 200_000
    STOP_RECALL = 0.999
    MIN_REPEAT_S = 0.5

SEED = 12345
