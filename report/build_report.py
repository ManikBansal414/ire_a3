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

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3 import config  # noqa: E402

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

    q = pd.read_csv(T / "part2_qps_at_recall.csv")
    for _, r in q.iterrows():
        key = f"{r.dataset}_{r['index'].split(' ')[0].replace('-', '').lower()}"
        n[f"{key}_qps90"] = r["QPS @0.90"]
        n[f"{key}_qps95"] = r["QPS @0.95"]
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
