#!/usr/bin/env bash
# Update the reference issue body so the next run picks up this run as
# its baseline. Best-effort: writes a warning and exits 0 on failure.

set -euo pipefail

: "${REFERENCE_ISSUE:?REFERENCE_ISSUE is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"

body=$(gh --repo "${GITHUB_REPOSITORY}" issue view "${REFERENCE_ISSUE}" \
    --json body -q .body 2>/dev/null || true)
if [[ -z "${body}" ]]; then
    echo "::warning::could not read issue #${REFERENCE_ISSUE}; skipping update"
    exit 0
fi

new_body=$(printf '%s\n' "${body}" \
    | sed "s/Inductor-XPU-OP-Benchmark-Data:.*/Inductor-XPU-OP-Benchmark-Data: ${GITHUB_RUN_ID}/" \
    | sed '/^$/d')

tmp=$(mktemp)
printf '%s\n' "${new_body}" > "${tmp}"
trap 'rm -f "${tmp}"' EXIT

if ! gh --repo "${GITHUB_REPOSITORY}" issue edit "${REFERENCE_ISSUE}" --body-file "${tmp}"; then
    echo "::warning::failed to update reference issue #${REFERENCE_ISSUE}"
fi
