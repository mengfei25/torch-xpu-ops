#!/usr/bin/env bash
# Download the previous microbench baseline artifact from the reference
# issue and stage it under $GITHUB_WORKSPACE/baseline. Tolerates missing
# baseline (first-ever run): emits a warning and exits 0 with no files.
#
# Inputs (env):
#   GH_TOKEN          GitHub token with issues:read permission
#   GITHUB_REPOSITORY  owner/repo
#   REFERENCE_ISSUE   issue number whose body has "Inductor-XPU-OP-Benchmark-Data: <run_id>"
#   ARTIFACT_DIR      where the upstream artifact was downloaded
#                     (default: $GITHUB_WORKSPACE/op_benchmark_tmp)
#   CURRENT_PREFIX    the current run's artifact prefix (we lift it out of ARTIFACT_DIR)

set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${REFERENCE_ISSUE:?REFERENCE_ISSUE is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

ARTIFACT_DIR="${ARTIFACT_DIR:-${GITHUB_WORKSPACE}/op_benchmark_tmp}"
CURRENT_PREFIX="${CURRENT_PREFIX:-Inductor-XPU-OP-Benchmark-Data}"

# 1. Move the current-run artifact into op_benchmark/.
latest_dir=$(find "${ARTIFACT_DIR}" -maxdepth 1 -type d \
    -name "${CURRENT_PREFIX}-*" 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -n "${latest_dir}" ]]; then
    mkdir -p "${GITHUB_WORKSPACE}/op_benchmark"
    mv "${latest_dir}"/* "${GITHUB_WORKSPACE}/op_benchmark/" 2>/dev/null || true
fi

# 2. Resolve the baseline run id from the reference issue body.
ref_run_id=$(gh --repo "${GITHUB_REPOSITORY}" issue view "${REFERENCE_ISSUE}" \
    --json body -q .body 2>/dev/null \
    | grep "Inductor-XPU-OP-Benchmark-Data" \
    | sed 's/.*: *//' \
    | head -n 1 || true)

mkdir -p "${GITHUB_WORKSPACE}/baseline"
if [[ -z "${ref_run_id}" ]]; then
    echo "::warning::no baseline run id in issue #${REFERENCE_ISSUE}; skipping comparison"
    exit 0
fi

# 3. Download the baseline run's artifacts.
work=$(mktemp -d)
pushd "${work}" >/dev/null
if ! gh --repo "${GITHUB_REPOSITORY}" run download "${ref_run_id}" \
        -p "Inductor-XPU-OP-Benchmark-Data-*" 2>/dev/null; then
    echo "::warning::failed to download baseline run ${ref_run_id}; skipping comparison"
    popd >/dev/null
    exit 0
fi
mkdir -p "${GITHUB_WORKSPACE}/reference"
shopt -s nullglob
matches=( Inductor-XPU-OP-Benchmark-Data-*-Updated )
shopt -u nullglob
if (( ${#matches[@]} > 0 )); then
    mv -f "${matches[0]}"/* "${GITHUB_WORKSPACE}/reference/"
else
    # Fall back: any matching dir.
    shopt -s nullglob
    any=( Inductor-XPU-OP-Benchmark-Data-* )
    shopt -u nullglob
    [[ ${#any[@]} -gt 0 ]] && mv -f "${any[0]}"/* "${GITHUB_WORKSPACE}/reference/"
fi
popd >/dev/null

# 4. Stage baseline_*.csv under baseline/.
ref="${GITHUB_WORKSPACE}/reference"
if [[ -f "${ref}/new_baseline/baseline_forward_op_summary.csv" ]]; then
    cp "${ref}/new_baseline/baseline_forward_op_summary.csv"  "${GITHUB_WORKSPACE}/baseline/"
    cp "${ref}/new_baseline/baseline_backward_op_summary.csv" "${GITHUB_WORKSPACE}/baseline/"
elif [[ -f "${ref}/forward_op_summary.csv" ]]; then
    cp "${ref}/forward_op_summary.csv"  "${GITHUB_WORKSPACE}/baseline/baseline_forward_op_summary.csv"
    cp "${ref}/backward_op_summary.csv" "${GITHUB_WORKSPACE}/baseline/baseline_backward_op_summary.csv"
else
    echo "::warning::no baseline summary CSV found in downloaded reference"
fi
