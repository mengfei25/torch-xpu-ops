#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

export PYTORCH_TEST_WITH_SLOW=1

ut::source_oneapi
LOG_DIR="$(ut::log_dir op_ut)"

cd "$(ut::ops_dir)/test/xpu"

python run_test_with_skip.py \
    2> "${LOG_DIR}/op_ut_with_skip_test_error.log" \
    | tee "${LOG_DIR}/op_ut_with_skip_test.log" || true

find . -name "op_ut_with_*.xml" -exec cp {} "${GITHUB_WORKSPACE}/ut_log/" \; 2>/dev/null || true

ut::reproduce op_ut "cd $(pwd) && pytest -sv <failed_case>"
