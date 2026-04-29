#!/usr/bin/env bash
# Collect runtime environment information into both a markdown table
# (intended for $GITHUB_STEP_SUMMARY) and a structured JSON file. Designed
# to be invoked from the `collect-environment` composite action but works
# standalone with the env vars below.
#
# Inputs (env):
#   OS_PACKAGES   space-separated extra dpkg packages to query
#   PIP_PACKAGES  space-separated extra pip modules to query
#   OUT_MD        markdown output (default: $GITHUB_STEP_SUMMARY)
#   OUT_JSON      JSON output (default: ./environment.json)
#   JOB_NAME      logical job name (default: $GITHUB_JOB)

set -euo pipefail

OUT_MD="${OUT_MD:-${GITHUB_STEP_SUMMARY:-/dev/stdout}}"
OUT_JSON="${OUT_JSON:-environment.json}"
JOB_NAME="${JOB_NAME:-${GITHUB_JOB:-job}}"
OS_PACKAGES="${OS_PACKAGES:-}"
PIP_PACKAGES="${PIP_PACKAGES:-}"

mkdir -p "$(dirname "${OUT_JSON}")"

# Defaults that every XPU environment should report on top of caller-provided extras.
DEFAULT_OS_PKGS="level-zero libigc1 libigc2 libze1 libze-intel-gpu1 intel-i915-dkms intel-level-zero-gpu intel-opencl-icd"
DEFAULT_PIP_PKGS="numpy torch torchaudio torchvision"

ALL_OS_PKGS="${OS_PACKAGES} ${DEFAULT_OS_PKGS}"
ALL_PIP_PKGS="${PIP_PACKAGES} ${DEFAULT_PIP_PKGS}"

dpkg_version() {
    dpkg -l 2>/dev/null | awk -v p="$1" '$1=="ii" && $2==p { print $3; exit }' || true
}
pip_version() {
    python -c "import $1; print($1.__version__)" 2>/dev/null || true
}
torch_attr() {
    python -c "import torch; print($1)" 2>/dev/null || true
}

# ---- collect ----
os_id=$( . /etc/os-release 2>/dev/null && echo "${VERSION_ID:-unknown}" || echo "unknown" )
linux_kernel=$(uname -r 2>/dev/null || echo "unknown")
python_ver=$(python --version 2>&1 | cut -f2 -d' ' || echo "unknown")
torch_xpu_version=$(torch_attr 'torch.version.xpu' || true)
xpu_count=$(torch_attr 'torch.xpu.device_count()' || true)

render_vendor_ids="[$(cat /sys/class/drm/render*/device/vendor 2>/dev/null | tr '\n' ' ' | sed 's/ /,/g' | sed 's/,$//')]"
render_device_ids="[$(cat /sys/class/drm/render*/device/device 2>/dev/null | tr '\n' ' ' | sed 's/ /,/g' | sed 's/,$//')]"

# ---- emit markdown ----
{
    echo "### Environment"
    echo "| | |"
    echo "| --- | --- |"
    echo "| jobs.${JOB_NAME}.versions.os | ${os_id} |"
    echo "| jobs.${JOB_NAME}.versions.linux-kernel | ${linux_kernel} |"
    echo "| jobs.${JOB_NAME}.versions.python | ${python_ver} |"
    for p in ${ALL_OS_PKGS}; do
        echo "| jobs.${JOB_NAME}.versions.${p} | $(dpkg_version "${p}") |"
    done
    for p in ${ALL_PIP_PKGS}; do
        echo "| jobs.${JOB_NAME}.versions.${p} | $(pip_version "${p}") |"
    done
    echo "| jobs.${JOB_NAME}.drm.render_nodes_vendor_ids | ${render_vendor_ids} |"
    echo "| jobs.${JOB_NAME}.drm.render_nodes_device_ids | ${render_device_ids} |"
    echo "| jobs.${JOB_NAME}.torch.version.xpu | ${torch_xpu_version} |"
    echo "| jobs.${JOB_NAME}.torch.xpu.device_count | ${xpu_count} |"
    echo "| jobs.${JOB_NAME}.env.ZE_AFFINITY_MASK | ${ZE_AFFINITY_MASK:-} |"
    echo "| jobs.${JOB_NAME}.env.NEOReadDebugKeys | ${NEOReadDebugKeys:-} |"
    echo "| jobs.${JOB_NAME}.env.PYTORCH_ENABLE_XPU_FALLBACK | ${PYTORCH_ENABLE_XPU_FALLBACK:-} |"
    echo "| jobs.${JOB_NAME}.env.PYTORCH_DEBUG_XPU_FALLBACK | ${PYTORCH_DEBUG_XPU_FALLBACK:-} |"
} >> "${OUT_MD}"

# ---- emit JSON ----
export OUT_JSON JOB_NAME ALL_OS_PKGS ALL_PIP_PKGS
export OS_ID="${os_id}"
export LINUX_KERNEL="${linux_kernel}"
export PYTHON_VER="${python_ver}"
export TORCH_XPU_VERSION="${torch_xpu_version}"
export XPU_COUNT="${xpu_count}"
export RENDER_VENDOR_IDS="${render_vendor_ids}"
export RENDER_DEVICE_IDS="${render_device_ids}"

python3 - <<'PY'
import json
import os
import subprocess

def dpkg_v(p):
    try:
        out = subprocess.check_output(["dpkg", "-l"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 3 and f[0] == "ii" and f[1] == p:
            return f[2]
    return ""

def pip_v(p):
    try:
        m = __import__(p)
        return getattr(m, "__version__", "")
    except Exception:
        return ""

data = {
    "job": os.environ["JOB_NAME"],
    "versions": {
        "os": os.environ.get("OS_ID", ""),
        "linux_kernel": os.environ.get("LINUX_KERNEL", ""),
        "python": os.environ.get("PYTHON_VER", ""),
    },
    "torch": {
        "xpu_version": os.environ.get("TORCH_XPU_VERSION", ""),
        "xpu_device_count": os.environ.get("XPU_COUNT", ""),
    },
    "drm": {
        "render_nodes_vendor_ids": os.environ.get("RENDER_VENDOR_IDS", ""),
        "render_nodes_device_ids": os.environ.get("RENDER_DEVICE_IDS", ""),
    },
    "env": {k: os.environ.get(k, "") for k in (
        "ZE_AFFINITY_MASK",
        "NEOReadDebugKeys",
        "PYTORCH_ENABLE_XPU_FALLBACK",
        "PYTORCH_DEBUG_XPU_FALLBACK",
    )},
    "os_packages": {p: dpkg_v(p) for p in os.environ["ALL_OS_PKGS"].split()},
    "pip_packages": {p: pip_v(p) for p in os.environ["ALL_PIP_PKGS"].split()},
}
with open(os.environ["OUT_JSON"], "w") as fh:
    json.dump(data, fh, indent=2, sort_keys=True)
PY
