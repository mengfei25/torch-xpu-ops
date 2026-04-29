#!/usr/bin/env bash
# Compare current microbench summary CSVs against the staged baseline,
# then update the baseline (best-of) for the next run. Skipped silently
# when the baseline is missing (first-ever run).

set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
SCRIPTS="${GITHUB_WORKSPACE}/.github/scripts"
WS="${GITHUB_WORKSPACE}"

pip install --quiet tabulate pandas

if [[ ! -f "${WS}/baseline/baseline_forward_op_summary.csv" ]]; then
    echo "::notice::no baseline staged - skipping comparison"
    exit 0
fi

echo "::group::Forward op regression check"
python "${SCRIPTS}/op_perf_comparison.py" \
    --xpu_file      "${WS}/op_benchmark/forward_op_summary.csv" \
    --baseline_file "${WS}/baseline/baseline_forward_op_summary.csv"
echo "::endgroup::"

echo "::group::Backward op regression check"
python "${SCRIPTS}/op_perf_comparison.py" \
    --xpu_file      "${WS}/op_benchmark/backward_op_summary.csv" \
    --baseline_file "${WS}/baseline/baseline_backward_op_summary.csv"
echo "::endgroup::"

# Update best-of baseline for the next run.
mkdir -p "${WS}/new_baseline"
cp "${WS}/baseline/"baseline*.csv "${WS}/new_baseline/"

python "${SCRIPTS}/op_calculate_best_perf.py" \
    --xpu      "${WS}/op_benchmark/forward_op_summary.csv" \
    --baseline "${WS}/new_baseline/baseline_forward_op_summary.csv" -r
python "${SCRIPTS}/op_calculate_best_perf.py" \
    --xpu      "${WS}/op_benchmark/backward_op_summary.csv" \
    --baseline "${WS}/new_baseline/baseline_backward_op_summary.csv" -r

cp -r "${WS}/new_baseline" "${WS}/op_benchmark/"
