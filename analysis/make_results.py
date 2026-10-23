"""Merge raw csvs into results/results.csv and draw every figure and table.

Selection rule used for every "at recall X" number: for each index type,
pick (build, search parameter) using only the *tune* half: within each build
the smallest parameter whose tune recall >= X, then the build whose tune
single-thread QPS is highest. The reported recall and QPS are those of that
choice on the *test* half.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3 import config  # noqa: E402

COLORS = {"HNSW": "#2a78d6", "IVF-Flat": "#eb6834", "IVF-PQ": "#1baf7a",
          "IVF-PQ+rerank": "#6250d6", "Exact": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#b7d3f2", "#86b5ea", "#4f93de", "#2a78d6", "#1b5aa6", "#0f3a6e"]
INDEX_ORDER = ["HNSW", "IVF-Flat", "IVF-PQ"]
TARGETS = [0.90, 0.95]

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 1.8, "lines.markersize": 3.5,
})


def load_raw():
    frames = {}
    for p in ["part1", "part2", "part3", "part4"]:
        f = config.RAW_DIR / f"{p}.csv"
        if f.exists():
            frames[p] = pd.read_csv(f)
    allrows = pd.concat(frames.values(), ignore_index=True, sort=False)
    allrows.to_csv(config.RESULTS_DIR / "results.csv", index=False)
    return frames


def select(df, target):
    """Tune-half selection described in the module docstring. Returns a row or None."""
    ok = df[df.recall_tune >= target]
    if ok.empty:
        return None
    firsts = ok.sort_values("param_value").groupby("build_params", as_index=False).first()
    return firsts.sort_values("qps_tune", ascending=False).iloc[0]


def pareto(df):
    """Points not beaten on both test recall and QPS, sorted by recall."""
    d = df.sort_values(["qps_test", "recall_test"], ascending=[False, False])
    keep, best = [], -1.0
    for _, r in d.iterrows():
        if r.recall_test > best:
            keep.append(r)
            best = r.recall_test
    return pd.DataFrame(keep).sort_values("recall_test")


def fmt_params(bp, extra=None):
    d = json.loads(bp)
    d.pop("efConstruction", None)
    d.pop("nbits", None)
    s = ", ".join(f"{k}={v}" for k, v in d.items())
    return s + (f", {extra}" if extra else "")


def write_table(name, df, caption=""):
    df.to_csv(config.TABLE_DIR / f"{name}.csv", index=False)
    html = df.to_html(index=False, border=0, classes="tbl", na_rep="–", escape=False)
    (config.TABLE_DIR / f"{name}.html").write_text(html, encoding="utf-8")
    (config.TABLE_DIR / f"{name}.md").write_text(
        (caption + "\n\n" if caption else "") + df.to_markdown(index=False), encoding="utf-8")


def exact_row(p1, ds, mode="one-at-a-time, 1 thread"):
    r = p1[(p1.dataset == ds) & (p1["mode"] == mode)]
    return r.iloc[0] if len(r) else None


# ---------------------------------------------------------------- Part 1
def part1_table(p1):
    with open(config.RAW_DIR / "part1_summary.json") as f:
        s = json.load(f)
    rows = []
    for ds in ["sift", "glove"]:
        sub = p1[p1.dataset == ds]
        chk = s[ds]["check_vs_stored"]
        r = {"dataset": ds, "N": int(sub.n_base.iloc[0])}
        for _, x in sub.iterrows():
            r[f"QPS {x['mode']}"] = round(x.qps_test, 1)
        one = exact_row(p1, ds)
        r["ms/query one-at-a-time"] = round(one.ms_per_query, 2)
        if isinstance(chk, dict):
            r["same top-10 id set as file"] = f"{chk['frac_queries_same_id_set']:.4f}"
            r["tie-aware recall of file lists"] = f"{chk['tie_aware_recall_of_stored_lists']:.4f}"
            r["max |dist diff|"] = f"{chk['max_abs_distance_diff']:.1e}"
        rows.append(r)
    write_table("part1_exact", pd.DataFrame(rows))


# ---------------------------------------------------------------- Part 2
def part2(p1, p2):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    qps_rows, cost_rows = [], []
    for ax, ds in zip(axes, ["sift", "glove"]):
        d = p2[p2.dataset == ds]
        for kind in INDEX_ORDER:
            sub = d[d["index"] == kind]
            ax.scatter(sub.recall_test, sub.qps_test, s=5, color=COLORS[kind], alpha=0.25, lw=0)
            pf = pareto(sub)
            ax.plot(pf.recall_test, pf.qps_test, "-o", color=COLORS[kind], label=kind)
        ex = exact_row(p1, ds)
        if ex is not None:
            ax.axhline(ex.qps_test, color=COLORS["Exact"], ls="--", lw=1.2)
            ax.plot([1.0], [ex.qps_test], "s", color=COLORS["Exact"], label="Exact (flat)")
        for t in TARGETS:
            ax.axvline(t, color=MUTED, lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xlim(0.25, 1.005)
        ax.set_xlabel("recall@10 (test half)")
        ax.set_title(f"{ds.upper() if ds == 'sift' else 'GloVe-100'} "
                     f"(N={int(d.n_base.iloc[0]):,})", color=INK, fontsize=9)
        ax.set_ylabel("QPS (1 thread, 1 query per call)")

        for kind in INDEX_ORDER:
            sub = d[d["index"] == kind]
            row = {"dataset": ds, "index": kind}
            for t in TARGETS:
                s = select(sub, t)
                if s is None:
                    row[f"QPS @{t:.2f}"] = f"not reached (max {sub.recall_test.max():.3f})"
                    row[f"setting @{t:.2f}"] = ""
                    row[f"dist/query @{t:.2f}"] = ""
                else:
                    row[f"QPS @{t:.2f}"] = f"{s.qps_test:,.0f}"
                    row[f"setting @{t:.2f}"] = (f"{fmt_params(s.build_params)}, "
                                               f"{s.param_name}={int(s.param_value)} "
                                               f"(test R={s.recall_test:.3f})")
                    row[f"dist/query @{t:.2f}"] = f"{s.ndis_per_query:,.0f}"
            qps_rows.append(row)
            s = select(sub, 0.95)
            if s is None:  # report the build that got closest, flagged
                best = sub.sort_values("recall_tune").iloc[-1]
                s, note = best, f"0.95 not reached; best build shown (max R={sub.recall_test.max():.3f})"
            else:
                note = ""
            cost_rows.append({
                "dataset": ds, "index": kind, "build": fmt_params(s.build_params),
                "build time (s)": round(s.build_time_s, 1),
                "of which training (s)": round(s.train_time_s, 1),
                "index size (MB)": round(s.index_bytes / 2 ** 20, 1),
                "bytes / vector": round(s.bytes_per_vector, 1),
                "note": note})
        if ex is not None:
            qps_rows.append({"dataset": ds, "index": "Exact (flat)",
                             "QPS @0.90": f"{ex.qps_test:,.1f}", "QPS @0.95": f"{ex.qps_test:,.1f}",
                             "setting @0.90": "recall 1.0", "setting @0.95": "recall 1.0",
                             "dist/query @0.90": f"{ex.n_base:,}", "dist/query @0.95": f"{ex.n_base:,}"})
            cost_rows.append({"dataset": ds, "index": "Exact (flat)", "build": "-",
                              "build time (s)": round(ex.build_time_s, 1), "of which training (s)": 0,
                              "index size (MB)": round(ex.index_bytes / 2 ** 20, 1),
                              "bytes / vector": round(ex.bytes_per_vector, 1), "note": ""})
    axes[1].set_ylabel("")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "part2_recall_vs_qps.png")
    plt.close(fig)
    write_table("part2_qps_at_recall", pd.DataFrame(qps_rows))
    write_table("part2_cost_at_095", pd.DataFrame(cost_rows))

    # all build settings: one figure per dataset, every build as its own curve
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    styles = ["-", "--", ":"]
    for ax, ds in zip(axes, ["sift", "glove"]):
        d = p2[p2.dataset == ds]
        for kind in INDEX_ORDER:
            for i, (bp, sub) in enumerate(d[d["index"] == kind].groupby("build_params")):
                sub = sub.sort_values("param_value")
                ax.plot(sub.recall_test, sub.qps_test, styles[i % 3], color=COLORS[kind],
                        lw=1.2, label=f"{kind} {fmt_params(bp)}")
        ax.set_yscale("log")
        ax.set_xlim(0.25, 1.005)
        ax.set_xlabel("recall@10 (test half)")
        ax.set_title(ds, color=INK, fontsize=9)
    axes[0].set_ylabel("queries / second")
    axes[1].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "part2_all_builds.png")
    plt.close(fig)


# ---------------------------------------------------------------- Part 3
def part3(p2, p3):
    ms = sorted(p3.m.unique())
    shades = dict(zip(ms, ["#a9cbf0", "#5b9be0", "#1f63b8", "#0b2a52", "#061a33"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for m in ms:
        for rr, ls in [(False, "--"), (True, "-")]:
            sub = p3[(p3.m == m) & (p3.rerank == rr)].sort_values("param_value")
            ax.plot(sub.recall_test, sub.qps_test, ls, marker="o", color=shades[m],
                    label=f"m={m}" + (" + rerank" if rr else ""))
    flat = p2[(p2.dataset == "sift") & (p2["index"] == "IVF-Flat")]
    nl = json.loads(p3.build_params.iloc[0])["nlist"]
    flat = flat[flat.build_params == json.dumps({"nlist": nl})].sort_values("param_value")
    ax.plot(flat.recall_test, flat.qps_test, "-", color=COLORS["IVF-Flat"], lw=2.2,
            label=f"IVF-Flat nlist={nl}")
    ax.set_yscale("log")
    ax.set_xlabel("recall@10 (test half)")
    ax.set_ylabel("queries / second")
    ax.set_title("IVF-PQ on SIFT: dashed = plain, solid = top-100 re-ranked", color=INK, fontsize=8.5)
    ax.legend(fontsize=6, ncol=2, loc="lower left")

    rows = []
    for m in ms:
        for rr in (False, True):
            sub = p3[(p3.m == m) & (p3.rerank == rr)]
            if sub.empty:
                continue
            row = {"m (bytes/code)": m, "re-rank top-100": "yes" if rr else "no",
                   "bytes / vector": round(sub.bytes_per_vector.iloc[0], 1),
                   "build (s)": round(sub.build_time_s.iloc[0], 1),
                   "max recall": round(sub.recall_test.max(), 3)}
            for t in TARGETS:
                s = select(sub, t)
                row[f"QPS @{t:.2f}"] = "–" if s is None else f"{s.qps_test:,.0f} (nprobe {int(s.param_value)})"
            rows.append(row)
    fr = {"m (bytes/code)": "IVF-Flat", "re-rank top-100": "-",
          "bytes / vector": round(flat.bytes_per_vector.iloc[0], 1),
          "build (s)": round(flat.build_time_s.iloc[0], 1), "max recall": round(flat.recall_test.max(), 3)}
    for t in TARGETS:
        s = select(flat, t)
        fr[f"QPS @{t:.2f}"] = "–" if s is None else f"{s.qps_test:,.0f} (nprobe {int(s.param_value)})"
    rows.append(fr)
    tbl = pd.DataFrame(rows)
    write_table("part3_compression", tbl)

    ax = axes[1]
    for rr, mk, lab in [(False, "o", "plain"), (True, "s", "+ rerank")]:
        sub = tbl[tbl["re-rank top-100"] == ("yes" if rr else "no")]
        ax.plot(sub["bytes / vector"], sub["max recall"], "-" if rr else "--", marker=mk,
                color=COLORS["IVF-PQ+rerank" if rr else "IVF-PQ"], label=f"IVF-PQ {lab}")
    lab_rows = tbl[(tbl["re-rank top-100"] == "no") |
                   ((tbl["re-rank top-100"] == "yes") & (tbl["max recall"] < 0.99))]
    top = tbl[(tbl["re-rank top-100"] == "yes") & (tbl["max recall"] >= 0.99)]
    if len(top):
        ax.annotate("m=" + ",".join(str(v) for v in top["m (bytes/code)"]),
                    (top["bytes / vector"].min(), 1.0), textcoords="offset points", xytext=(-8, -1),
                    ha="right", fontsize=6.5, color=MUTED)
    for _, r in lab_rows.iterrows():
        ax.annotate(f"m={r['m (bytes/code)']}", (r["bytes / vector"], r["max recall"]),
                    textcoords="offset points", xytext=(-4 if r["re-rank top-100"] == "yes" else 4, -3),
                    ha="right" if r["re-rank top-100"] == "yes" else "left", fontsize=6.5, color=MUTED)
    ax.plot([fr["bytes / vector"]], [fr["max recall"]], "D", color=COLORS["IVF-Flat"], label="IVF-Flat")
    ax.set_xscale("log")
    ax.set_xlabel("index bytes per vector (log)")
    ax.set_ylabel("best recall@10 reached")
    ax.set_title("memory vs recall ceiling", color=INK, fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "part3_compression.png")
    plt.close(fig)


# ---------------------------------------------------------------- Part 4
def part4(p4):
    if not (config.RAW_DIR / "part4_summary.json").exists():
        print("part4 not finished; skipping")
        return
    with open(config.RAW_DIR / "part4_summary.json") as f:
        summ = json.load(f)
    labels = [k for k in summ if not k.endswith("control")]
    shade = dict(zip(labels, BLUES[1:]))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, kind in zip(axes[:2], ["HNSW", "IVF-Flat"]):
        for lab in labels + ["d16-control"]:
            sub = p4[(p4.setting == lab) & (p4["index"] == kind)].sort_values("param_value")
            if sub.empty:
                continue
            ctrl = lab.endswith("control")
            ax.plot(sub.recall_test, sub.qps_test, ":" if ctrl else "-",
                    color=MUTED if ctrl else shade[lab], marker="" if ctrl else "o",
                    label=("d_int=16 (control, new seed)" if ctrl else f"d_int={summ[lab]['d_int']}"))
        ax.axvline(0.95, color=MUTED, lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xlim(0.3, 1.005)
        ax.set_xlabel("recall@10 (test half)")
        ax.set_title(f"{kind}", color=INK, fontsize=9)
    axes[0].set_ylabel("queries / second")
    axes[0].legend(fontsize=6, loc="lower left")

    rows = []
    for lab in labels + ["d16-control"]:
        st = summ[lab]
        r = {"setting": lab, "d_int": st["d_int"],
             "relative contrast": round(st["relative_contrast_median"], 3),
             "LID (MLE)": round(st["lid_mle_median"], 1),
             "exact ms/q": round(st["exact_ms_per_query"], 2)}
        for kind in ["HNSW", "IVF-Flat"]:
            sub = p4[(p4.setting == lab) & (p4["index"] == kind)]
            s = select(sub, 0.95)
            key = "efSearch" if kind == "HNSW" else "nprobe"
            if s is None:
                r[f"{kind} {key}@0.95"] = "not reached"
                r[f"{kind} QPS@0.95"] = "–"
                r[f"{kind} dist/query@0.95"] = np.nan
            else:
                r[f"{kind} {key}@0.95"] = int(s.param_value)
                r[f"{kind} QPS@0.95"] = round(s.qps_test)
                r[f"{kind} dist/query@0.95"] = round(s.ndis_per_query)
        rows.append(r)
    tbl = pd.DataFrame(rows)
    write_table("part4_intrinsic_dim", tbl)

    ax = axes[2]
    main = tbl[~tbl.setting.str.endswith("control")]
    n = int(p4.n_base.iloc[0])
    for kind in ["HNSW", "IVF-Flat"]:
        y = pd.to_numeric(main[f"{kind} dist/query@0.95"], errors="coerce")
        ax.plot(main.d_int, y / n, "-o", color=COLORS[kind], label=kind)
        c = tbl[tbl.setting.str.endswith("control")]
        ax.plot(c.d_int, pd.to_numeric(c[f"{kind} dist/query@0.95"], errors="coerce") / n, "x",
                color=COLORS[kind], ms=6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(main.d_int)
    ax.set_xticklabels(main.d_int)
    ax.set_xlabel("intrinsic dimension d_int (ambient 128)")
    ax.set_ylabel("fraction of base scanned for recall 0.95")
    ax.set_title("work needed (x = control)", color=INK, fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "part4_intrinsic_dim.png")
    plt.close(fig)


def benchmark_checks(frames):
    """Numbers for the 'could the benchmark mislead' question."""
    out = {}
    with open(config.RAW_DIR / "part1_summary.json") as f:
        s = json.load(f)
    out["python_call_overhead_us"] = s["python_call_overhead_us"]
    allr = pd.concat([frames[p] for p in ["part2", "part3", "part4"] if p in frames])
    # process CPU time on Windows ticks every ~15.6 ms: only judge long repeats
    long_r = allr[allr.repeat_seconds >= 1.0] if "repeat_seconds" in allr else allr
    out["cpu_wall_ratio_median"] = float(long_r.cpu_wall_ratio.median())
    out["cpu_wall_ratio_max"] = float(long_r.cpu_wall_ratio.max())
    out["cpu_wall_n_rows"] = int(len(long_r))
    reps = allr.qps_reps.astype(str).str.split(";").apply(lambda v: np.array(v, dtype=float))
    spread = reps.apply(lambda v: (v.max() - v.min()) / np.median(v))
    out["repeat_spread_median"] = float(spread.median())
    out["repeat_spread_p90"] = float(spread.quantile(0.9))
    diff = (allr.recall_test - allr.recall_tune)
    out["test_minus_tune_recall_mean"] = float(diff.mean())
    out["test_minus_tune_recall_absmax"] = float(diff.abs().max())
    out["timed_vs_batch_recall_absmax"] = float(
        (allr.recall_timed_subset - allr.recall_test).abs().max())
    p2 = frames["part2"]
    out["ms_per_query_min_part2"] = float(p2.ms_per_query.min())
    with open(config.TABLE_DIR / "benchmark_checks.json", "w") as f:
        json.dump(out, f, indent=2)


def main():
    frames = load_raw()
    if "part1" in frames:
        part1_table(frames["part1"])
    if "part2" in frames:
        part2(frames.get("part1"), frames["part2"])
    if "part3" in frames:
        part3(frames["part2"], frames["part3"])
    if "part4" in frames:
        part4(frames["part4"])
    benchmark_checks(frames)
    print("results.csv, figures and tables written to", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
