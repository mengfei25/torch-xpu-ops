#!/usr/bin/env bash
# Aggregate all transformers shard reports into a single GitHub step
# summary. Emits four sections:
#   1. Results table  (per-shard counts)
#   2. Baseline diff  (check-transformers.py)
#   3. Failure lines  (deduped failures across shards)
#   4. Not implemented ops (NotImplementedError + UserWarning patterns)
#
# Required env:
#   GITHUB_WORKSPACE
#   GITHUB_STEP_SUMMARY  (when running under GH Actions)
# Optional env:
#   REPORTS_DIR        default: $GITHUB_WORKSPACE/transformers/reports
#   LOGS_DIR           default: $GITHUB_WORKSPACE/transformers/logs
#   CHECK_SCRIPT       default: $GITHUB_WORKSPACE/torch-xpu-ops/.github/scripts/check-transformers.py

set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
REPORTS_DIR="${REPORTS_DIR:-${GITHUB_WORKSPACE}/transformers/reports}"
LOGS_DIR="${LOGS_DIR:-${GITHUB_WORKSPACE}/transformers/logs}"
CHECK_SCRIPT="${CHECK_SCRIPT:-${GITHUB_WORKSPACE}/torch-xpu-ops/.github/scripts/check-transformers.py}"

if [[ ! -d "${REPORTS_DIR}" ]]; then
    echo "::warning::no reports dir at ${REPORTS_DIR}; nothing to summarize"
    exit 0
fi

summary="${GITHUB_STEP_SUMMARY:-/dev/stdout}"

# Helper: parse counters from pytest's stats.txt summary line, e.g.
#   === 25 failed, 11 warnings, 0 errors ===
parse_stat() {
    local file="$1" key="$2" v
    v=$(grep -E "[0-9]+ ${key}\b" "${file}" 2>/dev/null \
        | sed -E "s/.* ([0-9]+) ${key}.*/\\1/" | head -n1 || true)
    [[ -n "${v}" ]] && echo "${v}" || echo "0"
}

# 1. Results table -----------------------------------------------------------
{
    echo "### Results"
    echo "| Test group | Errors | Failed | Deselected | Passed | Skipped |"
    echo "| --- | --- | --- | --- | --- | --- |"
    while IFS= read -r stat; do
        # reports/$test_group/stats.txt
        test_group=$(echo "${stat}" | cut -f $(($(echo "${REPORTS_DIR}" | tr -cd '/' | wc -c) + 2)) -d/)
        # Fall back: just take the parent directory name.
        test_group=$(basename "$(dirname "${stat}")")
        printf "| %s | %s | %s | %s | %s | %s |\n" \
            "${test_group}" \
            "$(parse_stat "${stat}" errors)" \
            "$(parse_stat "${stat}" failed)" \
            "$(parse_stat "${stat}" deselected)" \
            "$(parse_stat "${stat}" passed)" \
            "$(parse_stat "${stat}" skipped)"
    done < <(find "${REPORTS_DIR}" -name stats.txt)
} >> "${summary}"

# 2. Baseline diff -----------------------------------------------------------
if [[ -f "${CHECK_SCRIPT}" ]]; then
    {
        shopt -s nullglob
        xmls=( "${REPORTS_DIR}"/*.xml )
        shopt -u nullglob
        if (( ${#xmls[@]} > 0 )); then
            python "${CHECK_SCRIPT}" "${xmls[@]}" || true
        fi
    } >> "${summary}"
fi

# 3. Failure lines (deduped) -------------------------------------------------
{
    echo "### Failure lines"
    echo "| Test group | File | Error | Comment |"
    echo "| --- | --- | --- | --- |"

    fl_tmp=$(mktemp)
    trap 'rm -f "${fl_tmp}" "${fl_tmp}.uniq"' RETURN
    while IFS= read -r failure; do
        test_group=$(basename "$(dirname "${failure}")")
        # Drop header (first line); prefix each remaining line with the group.
        tail -n +2 "${failure}" | sed "s|^|${test_group} |" >> "${fl_tmp}"
    done < <(find "${REPORTS_DIR}" -name failures_line.txt)
    sort "${fl_tmp}" | uniq > "${fl_tmp}.uniq"
    while IFS= read -r line; do
        test_group=$(echo "${line}" | cut -f1 -d' ')
        file=$(echo "${line}"       | cut -f2 -d' ' | sed 's/\(.*\):$/\1/')
        error=$(echo "${line}"      | cut -f3 -d' ' | sed 's/\(.*\):$/\1/')
        comment="<pre>$(echo "${line}" | cut -f4- -d' ' | sed 's/\(.*\):$/\1/')</pre>"
        printf "| %s | %s | %s | %s |\n" "${test_group}" "${file}" "${error}" "${comment}"
    done < "${fl_tmp}.uniq"
} >> "${summary}"

# 4. Not-implemented XPU ops -------------------------------------------------
{
    echo "### Not implemented ops"
    echo "| Test group | Operator | Status |"
    echo "| --- | --- | --- |"

    ops_tmp=$(mktemp)
    trap 'rm -f "${ops_tmp}"' RETURN

    while IFS= read -r log; do
        test_group=$(basename "$(dirname "${log}")")
        while IFS= read -r op; do
            [[ -n "${op}" ]] && printf "| %s | <pre>%s</pre> | not implemented |\n" \
                "${test_group}" "${op}" >> "${ops_tmp}"
        done < <(grep NotImplementedError "${log}" 2>/dev/null \
                  | grep "for the XPU device" \
                  | sed "s/.*The operator '\(.*\)' is not.*/\1/")
    done < <(find "${REPORTS_DIR}" -name failures_line.txt)

    while IFS= read -r log; do
        test_group=$(basename "$(dirname "${log}")")
        while IFS= read -r op; do
            [[ -n "${op}" ]] && printf "| %s | <pre>%s</pre> | fallback to CPU happens |\n" \
                "${test_group}" "${op}" >> "${ops_tmp}"
        done < <(grep UserWarning "${log}" 2>/dev/null \
                  | grep "on the XPU backend" \
                  | sed "s/.*The operator '\(.*\) on the XPU.*/\1/")
    done < <(find "${REPORTS_DIR}" -name warnings.txt)

    sort -u "${ops_tmp}" || true
} >> "${summary}"

# 5. Environment dump (single, deduped) -------------------------------------
if [[ -d "${LOGS_DIR}" ]]; then
    first_md=$(find "${LOGS_DIR}" -name "environment-*.md" 2>/dev/null | head -n1 || true)
    if [[ -n "${first_md}" ]]; then
        cat "${first_md}" >> "${summary}"
        # Drop ZE_AFFINITY_MASK lines so cross-shard environment files match.
        find "${LOGS_DIR}" -name "environment-*.md" -print0 \
            | xargs -0 sed -i '/ZE_AFFINITY_MASK/d'
        while IFS= read -r f; do
            diff "${f}" "${first_md}" || true
        done < <(find "${LOGS_DIR}" -name "environment-*.md")
    fi
fi
