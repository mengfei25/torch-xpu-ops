#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

ut::source_oneapi
LOG_DIR="$(ut::log_dir xpu_profiling)"
mkdir -p "${LOG_DIR}/issue_reproduce"

cd "$(ut::ops_dir)"

# ResNet50 profiling
PROFILE=1 python -u test/profiling/rn50.py \
    -a resnet50 --dummy ./ --num-iterations 20 --xpu 0 || true
cp -f profiling.fp32.train.pt "${LOG_DIR}/" 2>/dev/null || true

# Issue reproduction tests
for t in \
    test/profiling/correlation_id_mixed.py \
    test/profiling/reproducer.missing.gpu.kernel.time.py \
    test/profiling/time_precision_in_profile.py \
    test/profiling/profile_partial_runtime_ops.py \
    test/profiling/triton_xpu_ops_time.py \
    test/profiling/test_profiler_correctness.py \
    test/profiling/test_for_overlapping_kernels.py; do
    if [[ -f "$t" ]]; then
        python -u "$t" | tee "${LOG_DIR}/issue_reproduce/$(basename "$t" .py).log" || true
    fi
done

# Llama profiling
pip install -q transformers || true
if [[ -f "test/profiling/llama.py" ]]; then
    python test/profiling/llama.py | tee "${LOG_DIR}/llama.log" || true
    if [[ -f ".github/scripts/llama_summary.py" ]]; then
        python .github/scripts/llama_summary.py \
            -i "${LOG_DIR}/llama.log" \
            -o "${LOG_DIR}/llama_summary.csv" || true
    fi
fi

# Upstream profiler tests
cd "${GITHUB_WORKSPACE}/pytorch/test/profiler"
for t in test_cpp_thread.py test_execution_trace.py test_memory_profiler.py \
         test_profiler_tree.py test_xpu_profiler.py; do
    if [[ -f "$t" ]]; then
        python -m pytest -s "$t" | tee "${LOG_DIR}/$(basename "$t" .py).log" || true
    fi
done
