#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

export PYTORCH_TEST_WITH_SLOW=1

ut::source_oneapi
LOG_DIR="$(ut::log_dir skipped_ut)"

cd "$(ut::ops_dir)/test/xpu"

# Skipped tests need a longer per-test timeout.
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS//--timeout 600/--timeout 3600}"
python run_test_with_skip.py --test-cases skipped \
    2> "${LOG_DIR}/skipped_ut_with_skip_test_error.log" \
    | tee "${LOG_DIR}/skipped_ut_with_skip_test.log" || true

find . -name "op_ut_with_*.xml" -exec cp {} "${GITHUB_WORKSPACE}/ut_log/" \; 2>/dev/null || true

ut::reproduce skipped_ut "cd $(pwd) && pytest -sv <failed_case>"
