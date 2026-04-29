#!/usr/bin/env bash
# Resolve the baseline run id for a given (build_kind, cadence) tuple by
# reading the body of $BASELINE_ISSUE_ID. Emits an empty value (warning)
# when the issue is unreachable or the line is missing.
#
# Required env:
#   GH_TOKEN
#   GITHUB_REPOSITORY
#   BASELINE_ISSUE_ID
#   ARTIFACT_TYPE         e.g. "build-nightly", "wheel-weekly"
# Optional env:
#   ISSUE_PREFIX          line prefix (default: "PVC -")
#   GITHUB_OUTPUT         when set, also writes run_id=<value>

set -euo pipefail

: "${BASELINE_ISSUE_ID:?BASELINE_ISSUE_ID is required}"
: "${ARTIFACT_TYPE:?ARTIFACT_TYPE is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

PREFIX="${ISSUE_PREFIX:-PVC -}"

run_id=""
if body=$(gh --repo "${GITHUB_REPOSITORY}" issue view "${BASELINE_ISSUE_ID}" \
                --json body -q .body 2>/dev/null); then
    run_id=$(printf '%s\n' "${body}" \
        | grep -i "^${PREFIX}${ARTIFACT_TYPE}:" \
        | sed 's/.*: *//' | head -n1 || true)
else
    echo "::warning::Failed to fetch issue ${BASELINE_ISSUE_ID}; skipping baseline."
fi

if [[ -z "${run_id}" ]]; then
    echo "::warning::No baseline found for ${ARTIFACT_TYPE} in issue #${BASELINE_ISSUE_ID}."
fi

echo "Resolved baseline run id: '${run_id}'"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "run_id=${run_id}" >> "${GITHUB_OUTPUT}"
fi
