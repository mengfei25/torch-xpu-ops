#!/usr/bin/env bash
# Aggregate per-suite Windows UT logs into a single working directory and
# run ut_result_check.sh for each suite name. Replaces the ~50-line
# inline summary_check block in _windows_ut.yml.
#
# Required env:
#   UT_NAMES        comma- or space-separated suite names (e.g. "op_ut,op_extended")
#   ARTIFACT_GLOB   glob for the per-shard artifact directories (e.g.
#                   "Inductor-XPU-UT-Data-1234-Windows-*")
#   UT_LOG_DIR      absolute path to the artifact root (default: $PWD/ut_log)
#   GH_TOKEN, GITHUB_REPOSITORY
# Optional env:
#   SCRIPTS_DIR     directory with fetch_known_issues.sh / ut_result_check.sh
#                   (default: this script's own directory)

set -euo pipefail

: "${UT_NAMES:?UT_NAMES is required}"
: "${ARTIFACT_GLOB:?ARTIFACT_GLOB is required}"

UT_LOG_DIR="${UT_LOG_DIR:-${PWD}/ut_log}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

ls -al "${UT_LOG_DIR}"
cd "${UT_LOG_DIR}"

# Pick the most recent matching shard directory and stage all loose log
# files into it so ut_result_check.sh sees a single flat layout.
latest_dir=$(find . -maxdepth 1 -type d -name "${ARTIFACT_GLOB}" | sort -V | tail -n1)
if [[ -z "${latest_dir}" ]]; then
    echo "::error::No artifact directory matched ${ARTIFACT_GLOB} under ${UT_LOG_DIR}"
    exit 1
fi
cd "${latest_dir}"

find "${UT_LOG_DIR}" -type f \
    \( -name "failures_*.log" -o -name "passed_*.log" -o -name "category_*.log" \) \
    -exec mv {} ./ \; 2>/dev/null || true

bash "${SCRIPTS_DIR}/fetch_known_issues.sh"
cp "${SCRIPTS_DIR}/ut_result_check.sh" ./

for ut_name in $(echo "${UT_NAMES}" | tr ',' ' '); do
    if [[ "${ut_name}" == "test_xpu" ]]; then
        echo "⏩ Skipping test_xpu check (temporarily disabled)"
        continue
    fi
    : > Known_issue.log
    if [[ -f "issues.log" ]]; then
        awk -v r="${ut_name}" 'BEGIN { print_row = 0 } {
            if ( ! ( $0 ~ /[a-zA-Z0-9]/ ) ) { print_row = 0 };
            if ( print_row == 1 && $1 ~ r ) { print $0 };
            if ( $0 ~ /Cases:/ ) { print_row = 1 };
        }' issues.log > Known_issue.log
    else
        echo "Info: issues.log not found or empty, using empty Known_issue.log"
    fi
    TEST_PLATFORM=windows bash ut_result_check.sh "${ut_name}"
done
