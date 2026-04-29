#!/usr/bin/env bash
# Run all microbenchmarks under pytorch/test/microbench and emit
# forward/backward summary CSVs.
#
# Inputs (env):
#   PYTORCH_DIR     pytorch source dir (default: $GITHUB_WORKSPACE/pytorch)
#   OUT_DIR         output dir for raw logs and summary CSVs
#                   (default: $GITHUB_WORKSPACE/op_benchmark)
#   SUMMARY_SCRIPT  path to microbench_summary.py
#                   (default: $GITHUB_WORKSPACE/.ci/scripts/summary/microbench_summary.py)

set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
PYTORCH_DIR="${PYTORCH_DIR:-${GITHUB_WORKSPACE}/pytorch}"
OUT_DIR="${OUT_DIR:-${GITHUB_WORKSPACE}/op_benchmark}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-${GITHUB_WORKSPACE}/.ci/scripts/summary/microbench_summary.py}"

mkdir -p "${OUT_DIR}"

bench_dir="${PYTORCH_DIR}/test/microbench"
[[ -d "${bench_dir}" ]] || bench_dir="test/microbench"
cd "${bench_dir}"

shopt -s nullglob
files=( *.py )
shopt -u nullglob
if (( ${#files[@]} == 0 )); then
    echo "::error::no microbench scripts under ${bench_dir}"
    exit 1
fi

echo "::group::Running ${#files[@]} microbench scripts"
for f in "${files[@]}"; do
    name="${f%.py}"
    echo "[microbench] ${name}"
    python "${f}" > "${OUT_DIR}/${name}.log" || \
        echo "::warning::microbench ${name} failed (continuing)"
done
echo "::endgroup::"

pip install --quiet pandas openpyxl

python "${SUMMARY_SCRIPT}" "${OUT_DIR}" "${OUT_DIR}/forward_op_summary.csv"
python "${SUMMARY_SCRIPT}" "${OUT_DIR}" "${OUT_DIR}/backward_op_summary.csv" --backward
