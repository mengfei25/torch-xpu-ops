#!/usr/bin/env bash
# Fetch open issues with skipped/skipped_windows labels and write them to
# issues.log in the current directory. Best-effort: any failure is a warning.
#
# Required env:
#   GH_TOKEN
#   GITHUB_REPOSITORY

set -euo pipefail

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

: > issues.log

count_linux=$(gh api "search/issues?q=repo:${GITHUB_REPOSITORY}+label:skipped+state:open" \
              --jq '.total_count' 2>/dev/null || echo 0)
if [[ "${count_linux}" =~ ^[0-9]+$ ]] && [[ "${count_linux}" -gt 0 ]]; then
    echo "${count_linux} issues with 'skipped' label found"
    gh api --paginate "repos/${GITHUB_REPOSITORY}/issues?labels=skipped" \
        --jq '.[] | select(.pull_request == null) | "Issue #\(.number): \(.title)\n\(.body)\n"' \
        >> issues.log || true
fi

count_windows=$(gh api "search/issues?q=repo:${GITHUB_REPOSITORY}+label:skipped_windows+state:open" \
                --jq '.total_count' 2>/dev/null || echo 0)
if [[ "${count_windows}" =~ ^[0-9]+$ ]] && [[ "${count_windows}" -gt 0 ]]; then
    echo "${count_windows} windows-only issues with 'skipped_windows' label found"
    gh api --paginate "repos/${GITHUB_REPOSITORY}/issues?labels=skipped_windows" \
        --jq '.[] | select(.pull_request == null) | "Issue #\(.number): \(.title)\n\(.body)\n"' \
        >> issues.log || true
fi
