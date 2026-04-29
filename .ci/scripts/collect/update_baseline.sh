#!/usr/bin/env bash
# Update the baseline-issue body so it points to a new run id for a
# given (artifact_type) line. Best-effort: warns and exits 0 on failure.
#
# Required env:
#   GH_TOKEN
#   GITHUB_REPOSITORY
#   BASELINE_ISSUE_ID
#   ARTIFACT_TYPE
#   NEW_RUN_ID
# Optional env:
#   ISSUE_PREFIX  default: "PVC -"

set -euo pipefail

: "${BASELINE_ISSUE_ID:?BASELINE_ISSUE_ID is required}"
: "${ARTIFACT_TYPE:?ARTIFACT_TYPE is required}"
: "${NEW_RUN_ID:?NEW_RUN_ID is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

PREFIX="${ISSUE_PREFIX:-PVC -}"
LINE="${PREFIX}${ARTIFACT_TYPE}"

body_file=$(mktemp)
trap 'rm -f "${body_file}"' EXIT

if ! gh --repo "${GITHUB_REPOSITORY}" issue view "${BASELINE_ISSUE_ID}" \
        --json body -q .body > "${body_file}" 2>/dev/null; then
    echo "::warning::Could not read issue #${BASELINE_ISSUE_ID}; skipping update."
    exit 0
fi

if grep -q "^${LINE}:" "${body_file}"; then
    sed -i "s|^${LINE}:.*|${LINE}: ${NEW_RUN_ID}|" "${body_file}"
else
    echo "${LINE}: ${NEW_RUN_ID}" >> "${body_file}"
fi

if ! gh --repo "${GITHUB_REPOSITORY}" issue edit "${BASELINE_ISSUE_ID}" \
        --body-file "${body_file}"; then
    echo "::warning::Failed to update issue #${BASELINE_ISSUE_ID}."
fi
