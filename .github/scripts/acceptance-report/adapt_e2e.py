#!/usr/bin/env python3
"""Reshape acceptance E2E CSVs into the gen_report.py schema.

Acceptance artifacts encode suite/dtype/mode in the *filename*
(e.g. inductor-huggingface-amp_fp16-training-xpu-accuracy.csv) with columns
`dev,name,batch_size,accuracy` / `dev,name,batch_size,speedup,abs_latency`.
gen_report.py instead expects suite/dtype/mode/scenario as *columns*. This script
copies each CSV into <out_dir> adding those columns (derived from the filename).

Usage: python adapt_e2e.py <src_dir> <out_dir>
Stdlib only.
"""

import csv
import os
import sys

SUITES = ["timm_models", "torchbench", "huggingface", "timm", "pt2e"]
DTYPES = ["amp_bf16", "amp_fp16", "bfloat16", "float16", "float32", "float64", "int8"]
MODES = ["training", "inference"]


def _find(base, needles):
    for n in sorted(needles, key=len, reverse=True):
        if n in base:
            return n
    return ""


def parse_meta(fname):
    """(suite, dtype, mode, result_type) from a filename; blanks when unknown."""
    base = os.path.basename(fname).lower()
    if base.endswith(".csv"):
        base = base[:-4]
    result = "accuracy" if "accuracy" in base else "performance"
    return _find(base, SUITES), _find(base, DTYPES), _find(base, MODES), result


def adapt_file(path, out_dir):
    suite, dtype, mode, result = parse_meta(path)
    metric = "accuracy" if result == "accuracy" else "speedup"
    out_cols = ["suite", "dtype", "mode", "name", "scenario", metric]
    out_name = f"inductor-{suite or 'unknown'}-{dtype or 'na'}-{mode or 'na'}-xpu-{result}.csv"
    out_path = os.path.join(out_dir, out_name)
    with open(path, newline="", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "suite": suite,
                "dtype": dtype,
                "mode": mode,
                "name": r.get("name", ""),
                "scenario": "",
                metric: r.get(metric, ""),
            })
    return out_path


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: adapt_e2e.py <src_dir> <out_dir>")
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if not os.path.isdir(src):
        print(f"adapt_e2e: source '{src}' missing, nothing to adapt")
        return
    count = 0
    for dirpath, _dirs, files in os.walk(src):
        for f in files:
            low = f.lower()
            if low.endswith("accuracy.csv") or low.endswith("performance.csv"):
                adapt_file(os.path.join(dirpath, f), out)
                count += 1
    print(f"adapt_e2e: reshaped {count} csv file(s) -> {out}")


if __name__ == "__main__":
    main()
