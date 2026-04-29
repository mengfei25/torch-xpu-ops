#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

export PYTORCH_TEST_WITH_SLOW=1
export PYTORCH_ROOT_DIR="${GITHUB_WORKSPACE}/pytorch"
export XML_OUTPUT_DIR="${GITHUB_WORKSPACE}/ut_log"
export XML_PREFIX=op_p0_with_

ut::source_oneapi
LOG_DIR="$(ut::log_dir op_p0)"

cd "${GITHUB_WORKSPACE}/pytorch/"

"${GITHUB_WORKSPACE}/.ci/benchmarks/p0/run_ut.sh" \
    2> "${LOG_DIR}/op_p0_with_skip_test_error.log" \
    | tee "${LOG_DIR}/op_p0_with_skip_test.log" || true

find . -name "op_p0_with_*.xml" -exec cp {} "${GITHUB_WORKSPACE}/ut_log/" \; 2>/dev/null || true

ut::reproduce op_p0 "cd $(pwd) && pytest -sv <failed_case>"
