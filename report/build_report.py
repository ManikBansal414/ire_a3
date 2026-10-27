"""Build report.pdf from report/template.html.

Placeholders in the template:
  {{fig:NAME}}    -> results/figures/NAME.png
  {{table:NAME}}  -> results/tables/NAME.html
  {{num:KEY}}     -> a number computed in numbers() from the result files
The filled HTML is printed to PDF with a headless Chromium browser (Edge or
Chrome, set BROWSER=/path/to/browser to override).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from a3 import config  # noqa: E402
from make_results import select  # noqa: E402  (same tune-half selection rule as the tables)

HERE = Path(__file__).resolve().parent
OUT_DIR = config.CODE_DIR.parent if not config.QUICK else config.RESULTS_DIR
OUT_HTML = OUT_DIR / ("report.html" if not config.QUICK else "report_quick.html")
OUT_PDF = OUT_DIR / ("report.pdf" if not config.QUICK else "report_quick.pdf")

BROWSERS = [
    os.environ.get("BROWSER", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "chromium", "chromium-browser", "google-chrome", "microsoft-edge",
]


def cpu_name():
    """Marketing name of the CPU (platform.processor() is vague on Windows)."""
    cmds = [["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
            ["sysctl", "-n", "machdep.cpu.brand_string"]]
    for c in cmds:
        try:
            out = subprocess.run(c, capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def numbers():
    """Every number quoted in the prose, keyed by name."""
    T = config.TABLE_DIR
    R = config.RAW_DIR
    n = {}
    with open(R / "part1_summary.json") as f:
        p1 = json.load(f)
    m = p1["machine"]
    n["machine"] = f"{cpu_name() or m['processor']}; {m['platform']}"
    n["versions"] = f"Python {m['python']}, FAISS {m['faiss']}, NumPy {m['numpy']}"
    n["threads"] = m["omp_max_threads"]
    p2 = pd.read_csv(R / "part2.csv")
    n["nlist_set"] = ", ".join(str(v) for v in sorted(
        {json.loads(b)["nlist"] for b in p2[p2["index"] == "IVF-Flat"].build_params}))
    n["n_timed"] = f"{int(p2.n_timed_queries.iloc[0]):,}"
    n["n_reps"] = int(p2.n_repeats.iloc[0])
    n["min_rep_s"] = config.MIN_REPEAT_S
    for ds in ["sift", "glove"]:
        d = p1[ds]
        n[f"{ds}_one_qps"] = f"{d['one_at_a_time_qps']:.0f}"
        n[f"{ds}_one_ms"] = f"{1000 / d['one_at_a_time_qps']:.1f}"
        n[f"{ds}_batch1_qps"] = f"{d['batch_1thread_qps']:,.0f}"
        n[f"{ds}_batchall_qps"] = f"{d['batch_allthreads_qps']:,.0f}"
        n[f"{ds}_batch_speedup"] = f"{d['batch_1thread_qps'] / d['one_at_a_time_qps']:.1f}"
        n[f"{ds}_batchall_speedup"] = f"{d['batch_allthreads_qps'] / d['one_at_a_time_qps']:.1f}"
        chk = d["check_vs_stored"]
        if isinstance(chk, dict):
            n[f"{ds}_same_set"] = f"{100 * chk['frac_queries_same_id_set']:.2f}%"
            n[f"{ds}_ndiff"] = chk["n_queries_id_set_differs"]
            n[f"{ds}_stored_recall"] = f"{chk['tie_aware_recall_of_stored_lists']:.4f}"
        else:
            n[f"{ds}_same_set"] = n[f"{ds}_ndiff"] = n[f"{ds}_stored_recall"] = "n/a (subset)"
        n[f"{ds}_overhead_us"] = f"{p1['python_call_overhead_us'][ds]:.1f}"
    with open(T / "benchmark_checks.json") as f:
        bc = json.load(f)
    n["cpu_wall_med"] = f"{bc['cpu_wall_ratio_median']:.2f}"
    n["cpu_wall_max"] = f"{bc['cpu_wall_ratio_max']:.2f}"
    n["rep_spread_med"] = f"{100 * bc['repeat_spread_median']:.1f}%"
    n["rep_spread_p90"] = f"{100 * bc['repeat_spread_p90']:.1f}%"
    n["tune_test_diff_mean"] = f"{bc['test_minus_tune_recall_mean']:+.4f}"
    n["tune_test_diff_max"] = f"{bc['test_minus_tune_recall_absmax']:.4f}"
    n["min_ms_part2"] = f"{1000 * bc['ms_per_query_min_part2']:.0f}"

    with open(R / "dataset_difficulty.json") as f:
        dd = json.load(f)
    for ds in ["sift", "glove"]:
        n[f"{ds}_rc"] = f"{dd[ds]['relative_contrast_median']:.2f}"
        n[f"{ds}_lid"] = f"{dd[ds]['lid_mle_median']:.0f}"
    q = pd.read_csv(T / "part2_qps_at_recall.csv")
    for _, r in q.iterrows():
        key = f"{r.dataset}_{r['index'].split(' ')[0].replace('-', '').lower()}"
        n[f"{key}_qps90"] = r["QPS @0.90"]
        n[f"{key}_qps95"] = r["QPS @0.95"]
    n.update(part2_numbers(R, p1))
    n.update(part3_numbers(R))
    n.update(part4_numbers(R))
    return n


def f2(x):
    return f"{x:.2f}"


def ratio(a, b):
    return f"{a / b:.1f}×"


def put(n, key, fn):
    """n[key] = fn(), or "n/a" when the result it needs does not exist (e.g. a
    recall target that a quick-mode run never reached)."""
    try:
        n[key] = fn()
    except Exception:  # noqa: BLE001
        n[key] = "n/a"


def part2_numbers(R, p1):
    n = {}
    d = pd.read_csv(R / "part2.csv")
    p1rows = pd.read_csv(R / "part1.csv")
    raw = {}
    for ds in ["sift", "glove"]:
        dd = d[d.dataset == ds]
        put(n, f"{ds}_pq_ceil", lambda: f2(dd[dd["index"] == "IVF-PQ"].recall_test.max()))
        for kind, key in [("HNSW", "hnsw"), ("IVF-Flat", "ivf")]:
            s = select(dd[dd["index"] == kind], 0.95)
            raw[(ds, key)] = s
            put(n, f"{ds}_{key}_q95", lambda: f"{s.qps_test:,.0f}")
            put(n, f"{ds}_{key}_p95", lambda: int(s.param_value))
            put(n, f"{ds}_{key}_b95", lambda: build_size(s))
        put(n, f"{ds}_ratio95", lambda: ratio(raw[(ds, "hnsw")].qps_test, raw[(ds, "ivf")].qps_test))
        put(n, f"{ds}_exact_qps", lambda: f"{p1[ds]['one_at_a_time_qps']:.0f}")
    put(n, "glove_slowdown",
        lambda: f"{raw[('sift', 'hnsw')].qps_test / raw[('glove', 'hnsw')].qps_test:.1f}×")

    # build cost / memory (Table 3 rows)
    c = pd.read_csv(config.TABLE_DIR / "part2_cost_at_095.csv")

    def g(ds, ix, col):
        return float(c[(c.dataset == ds) & (c["index"] == ix)][col].iloc[0])

    def bpv(ds, ix):
        return g(ds, ix, "bytes / vector")

    def rng(ix):
        a, b = bpv("sift", ix), bpv("glove", ix)
        return f"{min(a, b):.0f}–{max(a, b):.0f}"

    def savings():
        return [bpv(ds, ix) / bpv(ds, "IVF-PQ") for ds in ["sift", "glove"] for ix in ["HNSW", "IVF-Flat"]]

    pq_b = lambda: [g(ds, "IVF-PQ", "build time (s)") for ds in ["sift", "glove"]]  # noqa: E731
    put(n, "hnsw_build_sift", lambda: f"{g('sift', 'HNSW', 'build time (s)'):.0f}")
    put(n, "hnsw_build_glove", lambda: f"{g('glove', 'HNSW', 'build time (s)'):.0f}")
    n["hnsw_M_sift"] = n["sift_hnsw_b95"]
    n["hnsw_M_glove"] = n["glove_hnsw_b95"]
    put(n, "ivf_build", lambda: f"{np.mean([g(ds, 'IVF-Flat', 'build time (s)') for ds in ['sift', 'glove']]):.0f}")
    put(n, "ivf_train",
        lambda: f"{np.mean([g(ds, 'IVF-Flat', 'of which training (s)') for ds in ['sift', 'glove']]):.0f}")
    put(n, "pq_build", lambda: f"{min(pq_b()):.0f}–{max(pq_b()):.0f}")
    put(n, "hnsw_bpv", lambda: rng("HNSW"))
    put(n, "ivf_bpv", lambda: rng("IVF-Flat"))
    put(n, "pq_bpv", lambda: rng("IVF-PQ"))
    put(n, "pq_saving", lambda: f"{min(savings()):.0f}–{max(savings()):.0f}×")
    put(n, "pq_saving_sift_ivf", lambda: f"{bpv('sift', 'IVF-Flat') / bpv('sift', 'IVF-PQ'):.0f}×")
    put(n, "pq_mb_sift", lambda: f"{g('sift', 'IVF-PQ', 'index size (MB)'):.0f}")
    put(n, "full_mb_sift", lambda: (f"{g('sift', 'IVF-Flat', 'index size (MB)'):.0f}–"
                                    f"{g('sift', 'HNSW', 'index size (MB)'):.0f}"))

    # Q4 extrapolations from the measured exact cost (linear in N)
    one = p1rows[p1rows["mode"].astype(str).str.startswith("one-at-a-time")].set_index("dataset")

    def ms(ds):
        return float(one.loc[ds, "ms_per_query"])

    def nb(ds):
        return float(one.loc[ds, "n_base"])

    put(n, "sift_exact_ms", lambda: f"{ms('sift'):.0f}")
    put(n, "est150k_ms", lambda: f"{ms('sift') * 150_000 / nb('sift'):.1f}")
    put(n, "est150k_budget", lambda: f"{100 / (ms('sift') * 150_000 / nb('sift')):.0f}×")
    put(n, "est10m_sift", lambda: f"{ms('sift') * 1e7 / nb('sift'):.0f}")
    put(n, "est10m_glove", lambda: f"{ms('glove') * 1e7 / nb('glove'):.0f}")
    put(n, "hnsw10m_gb", lambda: f"{bpv('sift', 'HNSW') * 1e7 / 1e9:.1f}")
    put(n, "hnsw_bpv_sift", lambda: f"{bpv('sift', 'HNSW'):.0f}")
    put(n, "sift_ef95_x2", lambda: 2 * n["sift_hnsw_p95"])

    # Q5: per-call overhead relative to the fastest measured queries
    with open(R / "part1_summary.json") as f:
        ov = json.load(f)["python_call_overhead_us"]["sift"]

    def fastest95():
        return 1e6 / max(raw[("sift", "hnsw")].qps_test, raw[("glove", "hnsw")].qps_test)

    put(n, "fastest095_us", lambda: f"{fastest95():.0f}")
    put(n, "overhead_pct_095", lambda: f"{100 * ov / fastest95():.0f}%")
    put(n, "overhead_pct_fast", lambda: f"{100 * ov / (1000 * d.ms_per_query.min()):.0f}%")
    return n


def build_size(s):
    """M for HNSW, nlist for IVF, from a selected result row."""
    bp = json.loads(s.build_params)
    return bp.get("M", bp.get("nlist"))


def part3_numbers(R):
    n = {}
    p3 = pd.read_csv(R / "part3.csv")
    p2 = pd.read_csv(R / "part2.csv")
    nl = json.loads(p3.build_params.iloc[0])["nlist"]
    n["p3_nlist"] = nl
    flat = p2[(p2.dataset == "sift") & (p2["index"] == "IVF-Flat") &
              (p2.build_params == json.dumps({"nlist": nl}))]

    def sub(m, rr):
        return p3[(p3.m == m) & (p3.rerank == rr)]

    for m in [8, 16, 32, 64]:
        put(n, f"p3_plain_m{m}", lambda: f2(sub(m, False).recall_test.max()))
        put(n, f"p3_rr_m{m}", lambda: f"{sub(m, True).recall_test.max():.3f}")
    put(n, "p3_miss_m64", lambda: f"{100 * (1 - sub(64, False).recall_test.max()):.0f}%")
    small, large, maxdiff = [], [], 0.0
    fl = flat.set_index("param_value")
    for m in [8, 16, 32, 64]:
        a = sub(m, False).set_index("param_value").qps_test
        b = sub(m, True).set_index("param_value")
        for npr in a.index.intersection(b.index):
            c = 1 - b.qps_test[npr] / a[npr]   # fraction of QPS lost by re-ranking
            (small if npr <= 4 else large).append(c)
        if m >= 32:
            cm = b.index.intersection(fl.index)
            if len(cm):
                maxdiff = max(maxdiff, float((b.recall_test[cm] - fl.recall_test[cm]).abs().max()))
    put(n, "p3_cost_small", lambda: f"{100 * min(small):.0f}–{100 * max(small):.0f}%")
    put(n, "p3_cost_large_med", lambda: f"{100 * np.median(large):.0f}%")
    put(n, "p3_cost_large_max", lambda: f"{100 * max(large):.0f}%")
    n["p3_rr_flat_maxdiff"] = f"{maxdiff:.4f}"
    for m in [16, 32, 64]:
        put(n, f"p3_q95_m{m}", lambda: f"{select(sub(m, True), 0.95).qps_test:,.0f}")
    put(n, "p3_q95_flat", lambda: f"{select(flat, 0.95).qps_test:,.0f}")

    def beats_best_flat():
        # re-ranked m that beat the best IVF-Flat of Part 2 (any nlist) at recall 0.95
        best = select(p2[(p2.dataset == "sift") & (p2["index"] == "IVF-Flat")], 0.95).qps_test
        ms = [m for m in [8, 16, 32, 64]
              if select(sub(m, True), 0.95) is not None and select(sub(m, True), 0.95).qps_test > best]
        return "m = " + "/".join(map(str, ms)) if ms else "none"

    put(n, "p3_beats_best_flat", beats_best_flat)
    put(n, "p3_bpv_plain16", lambda: f"{sub(16, False).bytes_per_vector.iloc[0]:.1f}")
    put(n, "p3_bpv_rr16", lambda: f"{sub(16, True).bytes_per_vector.iloc[0]:.1f}")
    return n


def part4_numbers(R):
    n = {}
    p4 = pd.read_csv(R / "part4.csv")
    with open(R / "part4_summary.json") as f:
        summ = json.load(f)
    lab = ["d8", "d16", "d32", "d64", "d128"]
    N = int(p4.n_base.iloc[0])
    sel = {(l, k): select(p4[(p4.setting == l) & (p4["index"] == k)], 0.95)
           for l in lab + ["d16-control"] for k in ["HNSW", "IVF-Flat"]}
    h = [sel[(l, "HNSW")] for l in lab]
    v = [sel[(l, "IVF-Flat")] for l in lab]
    h_ok = [(l, s) for l, s in zip(lab, h) if s is not None]

    def steps():
        return [h_ok[i + 1][1].ndis_per_query / h_ok[i][1].ndis_per_query for i in range(len(h_ok) - 1)]

    def ctrl_diffs():
        return [abs(sel[("d16", k)].qps_test / sel[("d16-control", k)].qps_test - 1)
                for k in ["HNSW", "IVF-Flat"]]

    def ctrl_same():
        same = all(int(sel[("d16", k)].param_value) == int(sel[("d16-control", k)].param_value)
                   for k in ["HNSW", "IVF-Flat"])
        return "the same efSearch and nprobe" if same else "slightly different efSearch/nprobe"

    n["p4_n"] = f"{N:,}"
    put(n, "p4_nlist", lambda: json.loads(p4[p4["index"] == "IVF-Flat"].build_params.iloc[0])["nlist"])
    put(n, "p4_rc_d8", lambda: f"{summ['d8']['relative_contrast_median']:.1f}")
    put(n, "p4_rc_d128", lambda: f"{summ['d128']['relative_contrast_median']:.1f}")
    put(n, "p4_hnsw_last", lambda: h_ok[-1][0][1:])
    put(n, "p4_hnsw_ef", lambda: " → ".join(str(int(s.param_value)) for _, s in h_ok))
    put(n, "p4_hnsw_drop", lambda: f"{h_ok[0][1].qps_test / h_ok[-1][1].qps_test:.0f}×")
    put(n, "p4_hnsw_ndis_first", lambda: f"{h_ok[0][1].ndis_per_query:,.0f}")
    put(n, "p4_hnsw_ndis_last", lambda: f"{h_ok[-1][1].ndis_per_query:,.0f}")
    put(n, "p4_hnsw_ndis_ratio", lambda: f"{h_ok[-1][1].ndis_per_query / h_ok[0][1].ndis_per_query:.0f}")
    put(n, "p4_hnsw_step", lambda: f"{min(steps()):.1f}–{max(steps()):.1f}×")
    put(n, "p4_hnsw_max_d128",
        lambda: f2(p4[(p4.setting == "d128") & (p4["index"] == "HNSW")].recall_test.max()))
    put(n, "p4_ivf_np_d8", lambda: int(v[0].param_value))
    put(n, "p4_ivf_np_d128", lambda: int(v[-1].param_value))
    put(n, "p4_np_frac_d128", lambda: f"{100 * v[-1].param_value / n['p4_nlist']:.0f}%")
    put(n, "p4_ivf_ndis_ratio", lambda: f"{v[-1].ndis_per_query / v[0].ndis_per_query:.0f}")
    put(n, "p4_ivf_drop", lambda: f"{v[0].qps_test / v[-1].qps_test:.0f}×")
    put(n, "p4_ivf_drop_same", lambda: f"{v[0].qps_test / v[lab.index(h_ok[-1][0])].qps_test:.0f}×")
    n["p4_per_ds"] = f"{N + 10_000:,}"  # base + 10,000 queries per generated dataset
    put(n, "p4_ivf_q_d128", lambda: f"{v[-1].qps_test:,.0f}")
    put(n, "p4_ivf_frac_d128", lambda: f"{100 * v[-1].ndis_per_query / N:.0f}%")
    put(n, "p4_ivf_q_d64", lambda: f"{v[3].qps_test:,.0f}")
    put(n, "p4_hnsw_q_d64", lambda: f"{h[3].qps_test:,.0f}")
    put(n, "p4_ns_hnsw", lambda: f"{1e9 / h[3].qps_test / h[3].ndis_per_query:.0f}")
    put(n, "p4_hnsw_frac_d64", lambda: f"{100 * h[3].ndis_per_query / N:.0f}%")
    put(n, "p4_ns_ivf", lambda: f"{1e9 / v[3].qps_test / v[3].ndis_per_query:.0f}")
    put(n, "p4_ratio_d16", lambda: ratio(h[1].qps_test, v[1].qps_test))
    # control vs original d16
    ann = p4[p4["index"].isin(["HNSW", "IVF-Flat"])]  # exclude the exact-search rows
    a = ann[ann.setting == "d16"].set_index(["index", "param_value"])
    b = ann[ann.setting == "d16-control"].set_index(["index", "param_value"])
    j = a[["recall_test", "qps_test"]].join(b[["recall_test", "qps_test"]], rsuffix="_c", how="inner")
    put(n, "p4_ctrl_rec", lambda: f"{(j.recall_test - j.recall_test_c).abs().max():.3f}")
    put(n, "p4_ctrl_point_max", lambda: f"{100 * (j.qps_test / j.qps_test_c - 1).abs().max():.0f}%")
    put(n, "p4_ctrl_q95", lambda: f"{100 * min(ctrl_diffs()):.0f}–{100 * max(ctrl_diffs()):.0f}%")
    put(n, "p4_ctrl_same", ctrl_same)
    put(n, "p4_exact_range", lambda: (f"{min(summ[l]['exact_ms_per_query'] for l in lab):.1f}–"
                                      f"{max(summ[l]['exact_ms_per_query'] for l in lab):.1f}"))
    return n


def fill(template, nums):
    def rep(mo):
        kind, name = mo.group(1), mo.group(2)
        if kind == "fig":
            p = config.FIG_DIR / f"{name}.png"
            return f'<img src="{p.as_uri()}" alt="{name}">'
        if kind == "table":
            return (config.TABLE_DIR / f"{name}.html").read_text(encoding="utf-8")
        if name not in nums:
            print(f"warning: no number named {name}")
            return f"[{name}?]"
        return str(nums[name])
    return re.sub(r"\{\{(fig|table|num):([A-Za-z0-9_]+)\}\}", rep, template)


def to_pdf(html_path, pdf_path):
    for b in BROWSERS:
        exe = b if b and Path(b).exists() else (shutil.which(b) if b else None)
        if not exe:
            continue
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={pdf_path}",
                        html_path.as_uri()], check=True, capture_output=True, timeout=180)
        return True
    print("no Chromium-based browser found; open", html_path, "and print to PDF")
    return False


def main():
    nums = numbers()
    html = fill((HERE / "template.html").read_text(encoding="utf-8"), nums)
    html = html.replace("/*CSS*/", (HERE / "style.css").read_text(encoding="utf-8"))
    OUT_HTML.write_text(html, encoding="utf-8")
    if to_pdf(OUT_HTML, OUT_PDF):
        print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
