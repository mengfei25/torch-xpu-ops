#!/usr/bin/env bash
# Run a single transformers test shard and emit its JUnit XML.
#
# Required env:
#   TEST_CASE         identifier (used as report name)
#   TEST_CMD          pytest command suffix (target dirs + options)
#   TEST_FILTER       optional -k filter expression
#   TRANSFORMERS_DIR  transformers checkout (default: $GITHUB_WORKSPACE/transformers)

set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${TEST_CASE:?TEST_CASE is required}"
: "${TEST_CMD:?TEST_CMD is required}"

TRANSFORMERS_DIR="${TRANSFORMERS_DIR:-${GITHUB_WORKSPACE}/transformers}"
filter="${TEST_FILTER:-}"

cd "${TRANSFORMERS_DIR}"
mkdir -p reports

cmd=( python -m pytest --make-reports="${TEST_CASE}" --junit-xml="reports/${TEST_CASE}.xml" )
if [[ -n "${filter}" ]]; then
    cmd+=( -k "${filter}" )
fi
# shellcheck disable=SC2206
cmd+=( ${TEST_CMD} )

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### Running ${TEST_CASE}"
        echo '```'
        echo "${cmd[@]@Q}"
        echo '```'
    } >> "${GITHUB_STEP_SUMMARY}"
fi

# Always exit 0 so reporting steps still run; check-transformers.py decides
# the gating signal.
"${cmd[@]}" || true
