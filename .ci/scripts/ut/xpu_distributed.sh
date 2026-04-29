#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

ut::source_oneapi
LOG_DIR="$(ut::log_dir xpu_distributed)"

xpu-smi topology -m 2>/dev/null || true

cd "$(ut::ops_dir)/test/xpu"

# Verify XCCL availability before running.
XCCL=$(python -c "import torch; print(torch.distributed.is_xccl_available())" 2>/dev/null || echo False)
if [[ "${XCCL}" != "True" ]]; then
    echo "::error::XCCL not available"
    exit 1
fi

python run_distributed.py \
    2> "${LOG_DIR}/xpu_distributed_test_error.log" \
    | tee "${LOG_DIR}/xpu_distributed_test.log" || true

find .. -name "*.xml" -exec cp {} "${GITHUB_WORKSPACE}/ut_log/" \; 2>/dev/null || true
