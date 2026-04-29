#!/usr/bin/env bash
# Run the Accelerate XPU pytest suite with project-known exclusions and
# emit a JUnit XML at $REPORT_PATH.
#
# Inputs (env):
#   ACCELERATE_DIR   path to the accelerate checkout (default: $GITHUB_WORKSPACE/accelerate)
#   REPORT_PATH      JUnit XML output path (default: $ACCELERATE_DIR/reports/accelerate.xml)

set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
ACCELERATE_DIR="${ACCELERATE_DIR:-${GITHUB_WORKSPACE}/accelerate}"
REPORT_PATH="${REPORT_PATH:-${ACCELERATE_DIR}/reports/accelerate.xml}"

cd "${ACCELERATE_DIR}"
mkdir -p "$(dirname "${REPORT_PATH}")"

# Excluded tests:
#   * test_profiler                    - PTI_ERROR_INTERNAL on Kineto init for XPU
#   * test_gated                       - flaky env-config issue (HF gated repos)
#   * test_dispatch_model_..._offload  - OOM on huggingface/accelerate#3445
PATTERN="not test_profiler and not test_gated and not test_dispatch_model_tied_weights_memory_with_nested_offload_cpu"

cmd=(python -m pytest --junitxml="${REPORT_PATH}" -k "${PATTERN}" tests/)

# Echo the command into the step summary if available.
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### Running"
        echo '```'
        echo "${cmd[@]@Q}"
        echo '```'
    } >> "${GITHUB_STEP_SUMMARY}"
fi

"${cmd[@]}"
