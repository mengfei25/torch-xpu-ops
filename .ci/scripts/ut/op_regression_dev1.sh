#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

ut::source_oneapi
LOG_DIR="$(ut::log_dir op_regression_dev1)"

cd "$(ut::ops_dir)/test/regressions"

unset PYTEST_ADDOPTS ZE_AFFINITY_MASK
timeout 180 pytest -v test_operation_on_device_1.py \
    --junit-xml="${GITHUB_WORKSPACE}/ut_log/op_regression_dev1.xml" \
    2> "${LOG_DIR}/op_regression_dev1_test_error.log" \
    | tee "${LOG_DIR}/op_regression_dev1_test.log" || true

ut::reproduce op_regression_dev1 "cd $(pwd) && pytest -sv test_operation_on_device_1.py"
