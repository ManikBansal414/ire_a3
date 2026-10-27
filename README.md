# Assignment 3 — Benchmarking approximate nearest neighbour search

HNSW, IVF-Flat and IVF-PQ (FAISS) on SIFT-1M and GloVe-100, plus one
experiment on the standard data (Part 3: PQ compression and re-ranking) and
one on generated data (Part 4: intrinsic dimension).

## Layout

```
a3/            library code
  config.py      paths, full/quick mode settings
  data.py        HDF5 loading, GloVe normalisation, tune/test query split
  recall.py      recall@10 (ties at the 10th neighbour are not mistakes)
  timing.py      single-thread, one-query-at-a-time timing, median of repeats
  indexes.py     index builders (exact, HNSW, IVF-Flat, IVF-PQ, IVF-PQ+rerank)
  bench.py       ground truth, parameter sweeps, result rows
  synth.py       seeded generator for Part 4
experiments/   one script per part (part1_exact.py ... part4_intrinsic_dim.py)
analysis/      make_results.py: results.csv, figures, tables
report/        template.html + build_report.py -> report.pdf
tests/         unit tests (recall function, generator)
scripts/       download_data.py
hypothesis.md  Part 3 and Part 4 hypotheses (committed before the runs)
run_all.sh     one-command reproduction
```

## Setup

```
pip install -r requirements.txt
```

The two datasets (`sift-128-euclidean.hdf5`, `glove-100-angular.hdf5`, about
1 GB together) are downloaded from ann-benchmarks.com into `../data/` by the
first step of `run_all.sh` if they are not there already. Set `A3_DATA` to use
another folder.

## Reproduce

Run from inside `code/` with bash (Git Bash on Windows works):

```
bash run_all.sh            # full run: every number, figure and table in the report
bash run_all.sh --quick    # same pipeline on a small subset, < 15 min on a laptop
bash run_all.sh --only part3   # a single step: data, tests, part1, difficulty,
                               # part2, part3, part4, analyse, report
```

* **Full run** writes `../results/results.csv`, `../results/figures/`,
  `../results/tables/` and `../report.pdf`. It took 3 h 40 min on
  the machine below, almost all of it single-thread query
  timing (5,000 test queries x 3 repeats for every point of every sweep).
* **Quick run** uses the first 100,000 base vectors of each dataset (ground
  truth recomputed for that subset), 50,000 synthetic vectors in Part 4, scores
  recall on 2,000 queries from each half instead of 5,000, times 300 test
  queries (each timed repeat >= 0.1 s instead of 0.5 s), and stops sweeps at
  recall 0.99 instead of 0.999. It writes
  `../results_quick/` (including `report_quick.pdf`), so it never overwrites
  the full results. It took 8 min 48 s here (9 min 18 s on an earlier check), starting from an empty ground-truth cache.

Ground truth is cached in `../cache/` (delete it to recompute).

## Method in brief

* Queries 0–4,999 are the **tune** half and 5,000–9,999 the **test** half.
  Every choice (which build, which efSearch/nprobe reaches a recall target)
  is made from tune-half recall and tune-half timing; reported numbers are
  test-half numbers.
* Recall@10 is computed against our own exact FAISS search, from true
  distances recomputed in float64; a returned vector counts if it is no
  farther than the true 10th neighbour (so ties are not mistakes).
* Every measured number in the report text is a `{{num:...}}` placeholder that
  `report/build_report.py` computes from the result files (same tune-half
  selection rule as the tables), so a re-run updates text, tables and figures
  together. A value whose result does not exist in a run (e.g. a recall target
  the quick run never reaches) is printed as "n/a".
* QPS: one query per `search` call, `faiss.omp_set_num_threads(1)` plus
  `threadpoolctl` limiting BLAS to one thread, 50 warm-up queries, median of
  3 repeats over the timed queries (each repeat cycles through them for at
  least 0.5 s). CPU-time/wall-time is recorded to confirm
  single-thread execution. Index builds use all cores.
* A sweep also stops once a query costs more than exact search. That cap is
  exact search timed in one pass over 200 test queries (`bench.exact_ms`); it
  uses no recall, only speed. In Part 4 the same measurement is the "exact ms/q"
  of the report (rows with `index = Exact` in `results.csv`). It is a lighter
  measurement than the index timings (1 pass, 200 queries, not median of 3 over
  5,000); re-timing d_int = 8 and 16 the full way gave values within 6%.
* Index size is the length of `faiss.serialize_index`.
* The PDF is printed from HTML with headless Edge/Chrome (`BROWSER=...` to
  choose another Chromium browser); without one, open the generated
  `report.html` and print it.

## Machine

Intel Core Ultra 7 155U (12 cores / 14 threads), 32 GB RAM, Windows 11 Home
(10.0.26300), laptop on mains power. Python 3.12.4, faiss-cpu 1.15.0,
NumPy 2.3.4. Shell: Git Bash.

Library versions are pinned in `requirements.txt` and recorded in
`results/raw/part1_summary.json`.
