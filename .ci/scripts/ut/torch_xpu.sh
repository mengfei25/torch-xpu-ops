#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

export PYTORCH_TEST_WITH_SLOW=1
export PYTORCH_TESTING_DEVICE_ONLY_FOR=xpu

ut::source_oneapi
LOG_DIR="$(ut::log_dir torch_xpu)"

cd "${GITHUB_WORKSPACE}/pytorch"

# Collect all inductor + xpu test files
TEST_INCLUDES=""
for f in test/inductor/test_*.py; do
    [[ -f "$f" ]] && TEST_INCLUDES+=" inductor/$(basename "$f")"
done
for f in test/xpu/test_*.py; do
    [[ -f "$f" ]] && TEST_INCLUDES+=" xpu/$(basename "$f")"
done
[[ -f "test/test_xpu.py" ]] && TEST_INCLUDES+=" test_xpu.py"

# shellcheck disable=SC2086
python test/run_test.py --include ${TEST_INCLUDES} \
    2> "${LOG_DIR}/torch_xpu_test_error.log" \
    | tee "${LOG_DIR}/torch_xpu_test.log" || true
