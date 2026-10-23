#!/usr/bin/env bash
# Regenerate every number, figure and table in the report.
#
#   bash run_all.sh           full run  -> ../results/, ../report.pdf  (several hours)
#   bash run_all.sh --quick   first 100K vectors, 50K synthetic -> ../results_quick/ (< 15 min)
#
# Datasets are read from ../data (override with A3_DATA=/path/to/dir).
# Individual steps can be run with e.g.  bash run_all.sh --only part3
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${PYTHON:-}" ]]; then PY=$PYTHON
elif command -v python >/dev/null 2>&1; then PY=python
else PY=python3; fi
export A3_QUICK=0
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) export A3_QUICK=1 ;;
    --only) ONLY="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

step() {  # step <name> <command...>
  local name=$1; shift
  if [[ -z "$ONLY" || "$ONLY" == "$name" ]]; then
    echo "=================== $name ($(date +%H:%M:%S))"
    "$@"
  fi
}

step data    $PY scripts/download_data.py
step tests   $PY -m pytest -q tests
step part1   $PY experiments/part1_exact.py
step difficulty $PY experiments/dataset_difficulty.py
step part2   $PY experiments/part2_benchmark.py
step part3   $PY experiments/part3_compression.py
step part4   $PY experiments/part4_intrinsic_dim.py
step analyse $PY analysis/make_results.py
step report  $PY report/build_report.py
echo "done ($(date +%H:%M:%S))"
