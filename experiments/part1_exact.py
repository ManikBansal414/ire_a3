"""Part 1: exact baseline.

* exact top-10 for all 10,000 queries with a FAISS flat index (this is the
  reference for every recall number in the assignment);
* check it against the neighbours stored in the HDF5 file;
* time exact search one query at a time (single thread) and as one batch
  (single thread and all threads), median of repeats.
"""
import json
import os
import sys

import faiss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3 import config, data  # noqa: E402
from a3.bench import Queries, Recorder, base_row, ground_truth, log, machine_info  # noqa: E402
from a3.indexes import build_exact  # noqa: E402
from a3.recall import recall_at_k  # noqa: E402
from a3.timing import python_call_overhead_us, single_thread, time_batch, time_one_at_a_time  # noqa: E402

K = config.K


def check_against_stored(ds, gt_D, gt_I):
    """Compare our exact top-10 with the stored ann-benchmarks neighbours."""
    stored = ds["stored_nn"][:, :K]
    ours = gt_I[:, :K]
    same_set = np.mean([set(a) == set(b) for a, b in zip(ours, stored)])
    same_order = float(np.mean(np.all(ours == stored, axis=1)))
    # tie-aware: score the stored lists with our recall function
    r_stored = recall_at_k(stored, ds["base"], ds["queries"], ours, K)
    # distances: convert ours (squared L2) to the file's metric
    if ds["metric_in_file"] == "euclidean":
        ours_d = np.sqrt(np.maximum(gt_D[:, :K], 0))
    else:  # angular in the file = 1 - cos; for unit vectors ||a-b||^2 = 2 - 2cos
        ours_d = gt_D[:, :K] / 2
    max_abs_diff = float(np.abs(ours_d - ds["stored_d"][:, :K]).max())
    # where id sets differ, is it a tie at the 10th place?
    diff_q = [i for i, (a, b) in enumerate(zip(ours, stored)) if set(a) != set(b)]
    return {
        "frac_queries_same_id_set": float(same_set),
        "frac_queries_same_order": same_order,
        "n_queries_id_set_differs": len(diff_q),
        "tie_aware_recall_of_stored_lists": r_stored,
        "max_abs_distance_diff": max_abs_diff,
        "example_differing_queries": diff_q[:5],
        "example_kth_dist_ours_vs_stored": [
            (float(ours_d[i, K - 1]), float(ds["stored_d"][i, K - 1])) for i in diff_q[:5]],
    }


def main():
    rec = Recorder("part1")
    summary = {"machine": machine_info(), "quick": config.QUICK}
    summary["python_call_overhead_us"] = {}
    for name in ["sift", "glove"]:
        log(f"=== {name}")
        ds = data.load(name, config.N_BASE)
        xb, xq = ds["base"], ds["queries"]
        log(f"base {xb.shape}, queries {xq.shape}")
        gt_D, gt_I = ground_truth(name, xb, xq)
        info = {"n_base": len(xb)}
        if not ds["subset"]:
            info["check_vs_stored"] = check_against_stored(ds, gt_D, gt_I)
            log("check vs stored:", info["check_vs_stored"])
        else:
            info["check_vs_stored"] = "skipped: stored neighbours refer to the full base set"
        q = Queries(xb, xq, gt_I)
        built = build_exact(xb)
        row = base_row("part1", name, xb, built, q)

        with single_thread():
            one = time_one_at_a_time(built.index, q.time_test, K, config.N_REPEATS,
                                     min_seconds=config.MIN_REPEAT_S)
            batch1 = time_batch(built.index, q.test, K, config.N_REPEATS)
        batch_all = time_batch(built.index, q.test, K, config.N_REPEATS)
        n_t = len(q.time_test)
        r_one = recall_at_k(one["ids"], xb, q.time_test, q.gt_test[:n_t], K, q.kth_test[:n_t])
        common = dict(param_name="", param_value="", recall_tune=1.0, recall_test=1.0)
        rec.add(**row, **common, mode="one-at-a-time, 1 thread", recall_timed_subset=r_one,
                qps_test=one["qps"], ms_per_query=one["ms_per_query"],
                qps_reps=";".join(f"{x:.2f}" for x in one["qps_reps"]),
                cpu_wall_ratio=one["cpu_wall_ratio"], repeat_seconds=one["repeat_seconds"])
        rec.add(**row, **common, mode="batch, 1 thread", qps_test=batch1["qps"],
                ms_per_query=batch1["ms_per_query"],
                qps_reps=";".join(f"{x:.2f}" for x in batch1["qps_reps"]))
        rec.add(**row, **common, mode=f"batch, {faiss.omp_get_max_threads()} threads",
                qps_test=batch_all["qps"], ms_per_query=batch_all["ms_per_query"],
                qps_reps=";".join(f"{x:.2f}" for x in batch_all["qps_reps"]))
        log(f"exact one-at-a-time {one['qps']:.1f} qps ({one['ms_per_query']:.2f} ms), "
            f"batch 1 thread {batch1['qps']:.1f} qps, batch all threads {batch_all['qps']:.1f} qps, "
            f"cpu/wall {one['cpu_wall_ratio']:.2f}, recall of timed ids {r_one:.4f}")
        info.update(one_at_a_time_qps=one["qps"], batch_1thread_qps=batch1["qps"],
                    batch_allthreads_qps=batch_all["qps"])
        summary[name] = info
        summary["python_call_overhead_us"][name] = python_call_overhead_us(xb.shape[1], K)
    log("python call overhead (us):", summary["python_call_overhead_us"])
    with open(config.RAW_DIR / "part1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
