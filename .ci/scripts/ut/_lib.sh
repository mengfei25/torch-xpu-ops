#!/usr/bin/env bash
# Shared helpers for UT scripts.
# Each suite script is invoked with cwd=$GITHUB_WORKSPACE and inherits
# WORKSPACE, PYTEST_ADDOPTS, ZE_AFFINITY_MASK, XPU_ONEAPI_PATH, etc.

set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${WORKSPACE:=${GITHUB_WORKSPACE}}"

ut::source_oneapi() {
    if [[ -n "${XPU_ONEAPI_PATH:-}" ]]; then
        local env_sh="${WORKSPACE}/torch-xpu-ops/.github/scripts/env.sh"
        if [[ -f "${env_sh}" ]]; then
            # shellcheck disable=SC1090
            source "${env_sh}"
        elif [[ -f "${XPU_ONEAPI_PATH}/setvars.sh" ]]; then
            # shellcheck disable=SC1091
            source "${XPU_ONEAPI_PATH}/setvars.sh"
        fi
    fi
}

ut::log_dir() {
    local name="${1:?suite name required}"
    local dir="${GITHUB_WORKSPACE}/ut_log/${name}"
    mkdir -p "${dir}"
    printf '%s' "${dir}"
}

ut::reproduce() {
    local name="${1:?}" cmd="${2:?}"
    echo "Reproduce: ${cmd}" \
        | tee "${GITHUB_WORKSPACE}/ut_log/reproduce_${name}.log" >/dev/null
}

ut::ops_dir() {
    printf '%s/pytorch/third_party/torch-xpu-ops' "${GITHUB_WORKSPACE}"
}
