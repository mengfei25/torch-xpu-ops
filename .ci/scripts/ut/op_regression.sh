#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

ut::source_oneapi
LOG_DIR="$(ut::log_dir op_regression)"

cd "$(ut::ops_dir)/test/regressions"

pytest --junit-xml="${GITHUB_WORKSPACE}/ut_log/op_regression.xml" \
    2> "${LOG_DIR}/op_regression_test_error.log" \
    | tee "${LOG_DIR}/op_regression_test.log" || true

ut::reproduce op_regression "cd $(pwd) && pytest -sv <failed_case>"
