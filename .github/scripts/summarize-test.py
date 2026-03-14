#!/usr/bin/env python3
"""
Unified Test Analysis Tool

Subcommands:
  e2e   Compare inductor end‑to‑end performance/accuracy results
  ut    Compare unit test JUnit XML results (with optional GitHub issue integration)
  env   Generate environment summary from a combined log (collect_env + printenv)
  full  Run all three analyses and merge their markdown reports

Examples:
  python test_analyzer.py e2e -t target_dir -b baseline_dir -o comparison.xlsx --markdown report.md
  python test_analyzer.py ut -i "results/*.xml" -o test_comparison.xlsx --markdown
  python test_analyzer.py env --input combined.log --output summary.md
  python test_analyzer.py env --baseline baseline.log --target target.log --output comparison.md
  python test_analyzer.py full \\
      --e2e-target-dir target_e2e --e2e-baseline-dir baseline_e2e \\
      --ut-input "ut_results/*.xml" \\
      --env-baseline baseline_env.log --env-target target_env.log \\
      --output full_report.md
"""

import os
import sys
import argparse
import logging
import re
import glob
import json
import time
import concurrent.futures
import dataclasses
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

# Optional GitHub integration
try:
    from github import Github, Auth
    from github.Issue import Issue
    from github.Repository import Repository
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

# ----------------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------------

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

# ----------------------------------------------------------------------
# E2E mode (inductor performance/accuracy comparison)
# ----------------------------------------------------------------------

# ---------- E2E constants ----------
E2E_KNOWN_SUITES = {"huggingface", "timm_models", "torchbench"}
E2E_KNOWN_DATA_TYPES = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"}
E2E_KNOWN_MODES = {"inference", "training"}
E2E_PERFORMANCE_THRESHOLD = 0.1

E2E_SUMMARY_LEVELS = [
    ("Overall", [], 0),
    ("By Suite", ["suite"], 1),
    ("By Suite+DataType+Mode", ["suite", "data_type", "mode"], 2)
]

E2E_COLUMN_RENAME_MAP = {
    'data_type': 'dtype',
    'batch_size_target': 'bs_tgt',
    'batch_size_baseline': 'bs_bsl',
    'accuracy_target': 'acc_tgt',
    'accuracy_baseline': 'acc_bsl',
    'comparison_acc': 'cmp_acc',
    'inductor_target': 'ind_tgt',
    'inductor_baseline': 'ind_bsl',
    'eager_target': 'eag_tgt',
    'eager_baseline': 'eag_bsl',
    'inductor_ratio': 'ind_ratio',
    'eager_ratio': 'eag_ratio',
    'comparison_perf': 'cmp_perf',
    'comparison': 'cmp',
    'target passed': 'tgt_ps',
    'baseline passed': 'bsl_ps',
    'total': 'total',
    'target passrate': 'tgt_pass%',
    'baseline passrate': 'bsl_pass%',
    'New failed': 'new_fail',
    'New passed': 'new_pass',
    'inductor ratio': 'ind_ratio',
    'eager ratio': 'eag_ratio',
}

# ---------- E2E file discovery ----------
def e2e_find_result_files(root_dir):
    perf_files = glob.glob(os.path.join(root_dir, "**", "*_performance.csv"), recursive=True)
    acc_files = glob.glob(os.path.join(root_dir, "**", "*_accuracy.csv"), recursive=True)
    return perf_files + acc_files

def e2e_parse_filename(filepath):
    basename = os.path.basename(filepath)
    if not basename.endswith(".csv"):
        raise ValueError("Not a CSV file")
    base = basename[:-4]
    if not base.startswith("inductor_"):
        raise ValueError(f"Filename does not start with 'inductor_': {basename}")
    rest = base[len("inductor_"):]

    suite = None
    for s in sorted(E2E_KNOWN_SUITES, key=len, reverse=True):
        if rest.startswith(s + "_"):
            suite = s
            rest = rest[len(s) + 1:]
            break
    if suite is None:
        raise ValueError(f"Unknown suite in {basename}")

    parts = rest.split('_')
    mode_index = None
    for i, part in enumerate(parts):
        if part in E2E_KNOWN_MODES:
            mode_index = i
            break
    if mode_index is None:
        raise ValueError(f"Could not find mode (inference/training) in {basename}")
    mode = parts[mode_index]

    data_type = "_".join(parts[:mode_index])
    if data_type not in E2E_KNOWN_DATA_TYPES:
        print(f"Warning: Unknown data_type '{data_type}' in {basename}")

    if mode_index + 1 >= len(parts) or parts[mode_index + 1] != "xpu":
        raise ValueError(f"Missing 'xpu' after mode in {basename}")
    if mode_index + 2 >= len(parts):
        raise ValueError(f"Missing result type in {basename}")
    result_type = parts[mode_index + 2]
    if result_type not in ("accuracy", "performance"):
        raise ValueError(f"Result type not recognized in {basename}")

    return suite, data_type, mode, result_type

# ---------- E2E loading ----------
def e2e_merge_accuracy(records):
    pass_recs = [r for r in records if 'pass' in str(r['accuracy'])]
    if pass_recs:
        return pass_recs[0]
    fail_recs = [r for r in records if 'fail' in str(r['accuracy'])]
    if fail_recs:
        return fail_recs[0]
    return records[0]

def e2e_merge_performance(records):
    positive = [r for r in records if r['inductor'] > 0 and r['eager'] > 0]
    if positive:
        return min(positive, key=lambda r: (r['inductor'], r['eager']))
    if records:
        return max(records, key=lambda r: (r['inductor'], r['eager']))
    return None

def e2e_load_results(file_list, result_type_filter):
    raw_by_key = {}
    for fpath in file_list:
        try:
            suite, data_type, mode, res_type = e2e_parse_filename(fpath)
        except ValueError as e:
            print(f"Skipping {fpath}: {e}")
            continue
        if res_type != result_type_filter:
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue
        for _, row in df.iterrows():
            if row["dev"].strip() not in ['cpu', 'xpu', 'cuda']:
                continue
            if result_type_filter == "accuracy":
                record = {
                    "suite": suite,
                    "data_type": data_type,
                    "mode": mode,
                    "model": row["name"],
                    "batch_size": row["batch_size"],
                    "accuracy": row["accuracy"]
                }
            else:
                speedup = row.get("speedup")
                abs_latency = row.get("abs_latency")
                if pd.isna(speedup) or pd.isna(abs_latency):
                    print(f"Warning: Missing speedup/abs_latency for {suite}/{data_type}/{mode}/{row.get('name')} in {fpath}")
                    continue
                eager = speedup * abs_latency
                inductor = abs_latency
                record = {
                    "suite": suite,
                    "data_type": data_type,
                    "mode": mode,
                    "model": row["name"],
                    "batch_size": row["batch_size"],
                    "inductor": inductor,
                    "eager": eager
                }
            key = (suite, data_type, mode, row["name"])
            raw_by_key.setdefault(key, []).append(record)

    merged = []
    for key, rec_list in raw_by_key.items():
        if result_type_filter == "accuracy":
            merged.append(e2e_merge_accuracy(rec_list))
        else:
            m = e2e_merge_performance(rec_list)
            if m is not None:
                merged.append(m)
    return merged

# ---------- E2E merging ----------
def e2e_merge_accuracy(target_records, baseline_records):
    target_df = pd.DataFrame(target_records)
    baseline_df = pd.DataFrame(baseline_records)
    if target_df.empty and baseline_df.empty:
        return pd.DataFrame()
    merge_keys = ["suite", "data_type", "mode", "model"]
    merged = pd.merge(target_df, baseline_df, on=merge_keys, how="outer",
                      suffixes=("_target", "_baseline"), indicator=True)
    for col in ["batch_size_target", "batch_size_baseline"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')
    def compare_acc(row):
        if pd.isna(row.get("accuracy_target")) and pd.isna(row.get("accuracy_baseline")):
            return ""
        elif pd.isna(row.get("accuracy_target")):
            return "new_failed"
        elif pd.isna(row.get("accuracy_baseline")):
            return "new_passed"
        elif 'pass' not in row["accuracy_target"] and 'pass' in row["accuracy_baseline"]:
            return "new_failed"
        elif 'fail_accuracy' not in row["accuracy_target"] and 'fail_accuracy' in row["accuracy_baseline"]:
            return "new_failed"
        elif 'pass' in row["accuracy_target"] and 'pass' not in row["accuracy_baseline"]:
            return "new_passed"
        elif 'fail_accuracy' in row["accuracy_target"] and 'fail_accuracy' not in row["accuracy_baseline"]:
            return "new_passed"
        else:
            return "no_changed"
    merged["comparison_acc"] = merged.apply(compare_acc, axis=1)
    cols = ["suite", "data_type", "mode", "model",
            "batch_size_target", "accuracy_target",
            "batch_size_baseline", "accuracy_baseline",
            "comparison_acc"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[cols].sort_values(by=["suite", "data_type", "mode", "model"])

def e2e_merge_performance(target_records, baseline_records):
    target_df = pd.DataFrame(target_records)
    baseline_df = pd.DataFrame(baseline_records)
    if target_df.empty and baseline_df.empty:
        return pd.DataFrame()
    merge_keys = ["suite", "data_type", "mode", "model"]
    merged = pd.merge(target_df, baseline_df, on=merge_keys, how="outer",
                      suffixes=("_target", "_baseline"), indicator=True)
    for col in ["batch_size_target", "batch_size_baseline"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')
    mask = merged["inductor_target"].notna() & (merged["inductor_target"].astype(float) > 0)
    merged.loc[mask, "inductor_ratio"] = (
        merged.loc[mask, "inductor_baseline"].astype(float) /
        merged.loc[mask, "inductor_target"].astype(float)
    )
    mask = merged["eager_target"].notna() & (merged["eager_target"].astype(float) > 0)
    merged.loc[mask, "eager_ratio"] = (
        merged.loc[mask, "eager_baseline"].astype(float) /
        merged.loc[mask, "eager_target"].astype(float)
    )
    for col in ["inductor_target", "inductor_baseline", "eager_target", "eager_baseline"]:
        if col in merged.columns:
            merged[col] = merged[col].round(4)
    for col in ["inductor_ratio", "eager_ratio"]:
        if col in merged.columns:
            merged[col] = merged[col].round(3)
    def compare_perf(row):
        if pd.isna(row.get("inductor_target")) and pd.isna(row.get("inductor_baseline")):
            return ""
        elif row["inductor_ratio"] < 0 and row["inductor_baseline"] < 0:
            return ""
        elif pd.isna(row.get("inductor_target")) or row["inductor_ratio"] < 0:
            return "new_failed"
        elif pd.isna(row.get("inductor_baseline")) or row["inductor_baseline"] < 0:
            return "new_passed"
        elif row["inductor_ratio"] < 1 - E2E_PERFORMANCE_THRESHOLD or row["eager_ratio"] < 1 - E2E_PERFORMANCE_THRESHOLD:
            return "new_dropped"
        elif row["inductor_ratio"] > 1 + E2E_PERFORMANCE_THRESHOLD or row["eager_ratio"] > 1 + E2E_PERFORMANCE_THRESHOLD:
            return "new_improved"
        else:
            return "no_changed"
    merged["comparison_perf"] = merged.apply(compare_perf, axis=1)
    cols = ["suite", "data_type", "mode", "model",
            "batch_size_target", "inductor_target", "eager_target",
            "batch_size_baseline", "inductor_baseline", "eager_baseline",
            "inductor_ratio", "eager_ratio", "comparison_perf"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[cols].sort_values(by=["suite", "data_type", "mode", "model"])

def e2e_combine_results(acc_merged, perf_merged):
    acc_renamed = None
    perf_renamed = None
    if not acc_merged.empty:
        acc_renamed = acc_merged.rename(columns={
            "batch_size_target": "batch_size_accuracy_target",
            "batch_size_baseline": "batch_size_accuracy_baseline"
        })
    if not perf_merged.empty:
        perf_renamed = perf_merged.rename(columns={
            "batch_size_target": "batch_size_performance_target",
            "batch_size_baseline": "batch_size_performance_baseline"
        })
    if acc_renamed is not None and perf_renamed is not None:
        merge_keys = ["suite", "data_type", "mode", "model"]
        combined = pd.merge(acc_renamed, perf_renamed, on=merge_keys, how="outer")
        def compare_result(row):
            if pd.isna(row.get("comparison_acc")) and pd.isna(row.get("comparison_perf")):
                return ""
            elif pd.isna(row.get("comparison_acc")):
                return row.get("comparison_perf")
            elif pd.isna(row.get("comparison_perf")):
                return row.get("comparison_acc")
            elif ('new_dropped' in [row.get("comparison_acc"), row.get("comparison_perf")] or
                  'new_failed' in [row.get("comparison_acc"), row.get("comparison_perf")]):
                return "new_failed"
            elif ('new_improved' in [row.get("comparison_acc"), row.get("comparison_perf")] or
                  'new_passed' in [row.get("comparison_acc"), row.get("comparison_perf")]):
                return "new_improved"
            else:
                return "no_changed"
        combined["comparison"] = combined.apply(compare_result, axis=1)
        return combined.sort_values(by=merge_keys)
    elif acc_renamed is not None:
        return acc_renamed.sort_values(by=["suite", "data_type", "mode", "model"])
    elif perf_renamed is not None:
        return perf_renamed.sort_values(by=["suite", "data_type", "mode", "model"])
    else:
        return pd.DataFrame()

# ---------- E2E summary ----------
def e2e_accuracy_metrics(group):
    def is_acc_pass(val):
        return pd.notna(val) and str(val) != "" and 'pass' in str(val)
    return pd.Series({
        'target_passed': group['accuracy_target'].apply(is_acc_pass).sum(),
        'baseline_passed': group['accuracy_baseline'].apply(is_acc_pass).sum(),
        'total': len(group),
        'new_failed': (group['comparison_acc'] == 'new_failed').sum(),
        'new_passed': (group['comparison_acc'] == 'new_passed').sum(),
    })

def e2e_performance_metrics(group):
    def is_perf_pass(val):
        return pd.notna(val) and str(val) != "" and int(val) > 0
    def geomean(series):
        vals = series.replace("", np.nan).replace(0, np.nan).dropna()
        if len(vals) == 0:
            return np.nan
        return np.exp(np.log(vals).mean())
    return pd.Series({
        'target_passed': group['inductor_target'].apply(is_perf_pass).sum(),
        'baseline_passed': group['inductor_baseline'].apply(is_perf_pass).sum(),
        'total': len(group),
        'new_failed': ((group['comparison_perf'] == 'new_failed') | (group['comparison_perf'] == 'new_dropped')).sum(),
        'new_passed': ((group['comparison_perf'] == 'new_passed') | (group['comparison_perf'] == 'new_improved')).sum(),
        'inductor_ratio_geomean': geomean(group['inductor_ratio']),
        'eager_ratio_geomean': geomean(group['eager_ratio']),
    })

def e2e_compute_group_summary(acc_merged, perf_merged, group_cols, level_name):
    summaries = []
    if not acc_merged.empty:
        if not group_cols:
            acc_sum = acc_merged.assign(_dummy='Overall').groupby('_dummy').apply(e2e_accuracy_metrics).reset_index(drop=True)
            acc_sum['Category'] = 'Overall'
        else:
            acc_sum = acc_merged.groupby(group_cols).apply(e2e_accuracy_metrics).reset_index()
            acc_sum['Category'] = acc_sum[group_cols].astype(str).agg('_'.join, axis=1)
        acc_sum['Type'] = 'Accuracy'
        acc_sum['Level'] = level_name
        summaries.append(acc_sum)
    if not perf_merged.empty:
        if not group_cols:
            perf_sum = perf_merged.assign(_dummy='Overall').groupby('_dummy').apply(e2e_performance_metrics).reset_index(drop=True)
            perf_sum['Category'] = 'Overall'
        else:
            perf_sum = perf_merged.groupby(group_cols).apply(e2e_performance_metrics).reset_index()
            perf_sum['Category'] = perf_sum[group_cols].astype(str).agg('_'.join, axis=1)
        perf_sum['Type'] = 'Performance'
        perf_sum['Level'] = level_name
        summaries.append(perf_sum)
    if not summaries:
        return pd.DataFrame()
    combined = pd.concat(summaries, ignore_index=True, sort=False)
    combined['target passrate'] = combined['target_passed'] / combined['total']
    combined['baseline passrate'] = combined['baseline_passed'] / combined['total']
    combined.rename(columns={
        'target_passed': 'target passed',
        'baseline_passed': 'baseline passed',
        'new_failed': 'New failed',
        'new_passed': 'New passed',
        'inductor_ratio_geomean': 'inductor ratio',
        'eager_ratio_geomean': 'eager ratio'
    }, inplace=True)
    cols = ['Level', 'Type', 'Category', 'target passed', 'baseline passed', 'total',
            'target passrate', 'baseline passrate', 'New failed', 'New passed',
            'inductor ratio', 'eager ratio']
    for col in cols:
        if col not in combined.columns:
            combined[col] = np.nan
    return combined[cols]

def e2e_generate_all_summaries(acc_merged, perf_merged):
    all_summaries = []
    for level_name, group_cols, priority in E2E_SUMMARY_LEVELS:
        df = e2e_compute_group_summary(acc_merged, perf_merged, group_cols, level_name)
        if not df.empty:
            df['SortPriority'] = priority
            all_summaries.append(df)
    if not all_summaries:
        return pd.DataFrame()
    final = pd.concat(all_summaries, ignore_index=True, sort=False)
    for col in ['target passed', 'baseline passed', 'total', 'New failed', 'New passed']:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors='coerce').astype('Int64')
    for col in ['target passrate', 'baseline passrate']:
        if col in final.columns:
            final[col] = (final[col] * 100).round(2)
    for col in ['inductor ratio', 'eager ratio']:
        if col in final.columns:
            final[col] = final[col].round(3)
    final.sort_values(['SortPriority', 'Type', 'Category'], inplace=True)
    final.drop(columns=['Level', 'SortPriority'], inplace=True, errors='ignore')
    return final.reset_index(drop=True)

# ---------- E2E markdown (returns string) ----------
def e2e_generate_markdown_string(combined_summary, details):
    """Return markdown string; empty if no data."""
    if combined_summary.empty and details.empty:
        return ""

    lines = []
    lines.append(f"# Inductor Test Results Comparison: Target vs Baseline\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ----- Overview by Suite -----
    if not details.empty and 'suite' in details.columns:
        suite_rows = []
        for suite in details['suite'].dropna().unique():
            suite_df = details[details['suite'] == suite]
            acc_fail = suite_df['cmp_acc'].eq('new_failed').sum() if 'cmp_acc' in suite_df else 0
            acc_pass = suite_df['cmp_acc'].eq('new_passed').sum() if 'cmp_acc' in suite_df else 0
            perf_fail = suite_df['cmp_perf'].eq('new_failed').sum() if 'cmp_perf' in suite_df else 0
            perf_drop = suite_df['cmp_perf'].eq('new_dropped').sum() if 'cmp_perf' in suite_df else 0
            perf_pass = suite_df['cmp_perf'].eq('new_passed').sum() if 'cmp_perf' in suite_df else 0
            perf_improve = suite_df['cmp_perf'].eq('new_improved').sum() if 'cmp_perf' in suite_df else 0
            suite_rows.append({
                'suite': suite,
                'acc_fail': acc_fail,
                'acc_pass': acc_pass,
                'perf_fail': perf_fail,
                'perf_drop': perf_drop,
                'perf_pass': perf_pass,
                'perf_improve': perf_improve
            })
        if suite_rows:
            lines.append("## 📊 Overview by Suite\n")
            def fmt_count(val, good=False):
                if val > 0:
                    emoji = "🟢" if good else "🔴"
                    return f"{val} {emoji}"
                return str(val)
            lines.append("| Suite | 🧪 Acc Fail | 🧪 Acc Pass | ⏱️ Perf Fail | ⏱️ Perf Drop | ⏱️ Perf Pass | ⏱️ Perf Improve |")
            lines.append("|-------|-------------|-------------|---------------|---------------|---------------|-----------------|")
            for s in suite_rows:
                row = [
                    s['suite'],
                    fmt_count(s['acc_fail'], good=False),
                    fmt_count(s['acc_pass'], good=True),
                    fmt_count(s['perf_fail'], good=False),
                    fmt_count(s['perf_drop'], good=False),
                    fmt_count(s['perf_pass'], good=True),
                    fmt_count(s['perf_improve'], good=True)
                ]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # ----- Overall Summary -----
    overall = combined_summary[combined_summary['Category'] == 'Overall']
    if not overall.empty:
        lines.append("## 📈 Overall Summary\n")
        def fmt_new(val, is_fail=True):
            if pd.notna(val) and val > 0:
                emoji = "🔴" if is_fail else "🟢"
                return f"{val} {emoji}"
            return str(val) if pd.notna(val) else ""
        def fmt_ratio(val):
            if pd.notna(val):
                if val < 0.95:
                    return f"{val} 🔴"
                elif val > 1.05:
                    return f"{val} 🟢"
            return str(val) if pd.notna(val) else ""
        lines.append("| Type | tgt_ps | bsl_ps | total | new_fail | new_pass | tgt_pass% | bsl_pass% | ind_ratio | eag_ratio |")
        lines.append("|------|--------|--------|-------|----------|----------|-----------|-----------|-----------|-----------|")
        for _, row in overall.iterrows():
            type_label = "Accuracy" if row['Type'] == 'Accuracy' else "Performance"
            tgt_ps = row.get('tgt_ps', '')
            bsl_ps = row.get('bsl_ps', '')
            total = row.get('total', '')
            new_fail = fmt_new(row.get('new_fail'), is_fail=True)
            new_pass = fmt_new(row.get('new_pass'), is_fail=False)
            tgt_pass = row.get('tgt_pass%', '')
            bsl_pass = row.get('bsl_pass%', '')
            ind_ratio = fmt_ratio(row.get('ind_ratio'))
            eag_ratio = fmt_ratio(row.get('eag_ratio'))
            lines.append(f"| {type_label} | {tgt_ps} | {bsl_ps} | {total} | {new_fail} | {new_pass} | {tgt_pass} | {bsl_pass} | {ind_ratio} | {eag_ratio} |")
        lines.append("")

    if details.empty:
        lines.append("No detailed data available.\n")
        return "\n".join(lines)

    # ----- Helper to generate an HTML table with row background colors -----
    def write_html_table(rows, columns, condition_column, fail_color="#f8d7da", pass_color="#d4edda"):
        table_lines = []
        table_lines.append('<table>')
        table_lines.append('<thead><tr>')
        for col in columns:
            table_lines.append(f'<th>{col}</th>')
        table_lines.append('</tr></thead>')
        table_lines.append('<tbody>')
        for _, row in rows.iterrows():
            val = row.get(condition_column, '')
            bg_color = ''
            if val in ['new_failed', 'new_dropped']:
                bg_color = f' style="background-color: {fail_color};"'
            elif val in ['new_passed', 'new_improved']:
                bg_color = f' style="background-color: {pass_color};"'
            table_lines.append(f'<tr{bg_color}>')
            for col in columns:
                cell = str(row.get(col, ''))
                table_lines.append(f'<td>{cell}</td>')
            table_lines.append('</tr>')
        table_lines.append('</tbody>')
        table_lines.append('</table>')
        return table_lines

    # ----- New Failures & Regressions -----
    acc_fail = details[details['cmp_acc'] == 'new_failed']
    perf_regress = details[details['cmp_perf'].isin(['new_dropped', 'new_failed'])]
    if not acc_fail.empty or not perf_regress.empty:
        lines.append("## ❌ New Failures & Regressions\n")
        if not acc_fail.empty:
            lines.append("### 🧪 Accuracy Failures\n")
            cols = ['suite', 'dtype', 'mode', 'model', 'bs_acc_tgt', 'acc_tgt', 'bs_acc_bsl', 'acc_bsl', 'cmp_acc']
            available = [c for c in cols if c in acc_fail.columns]
            lines.extend(write_html_table(acc_fail, available, 'cmp_acc', fail_color="#f8d7da"))
            lines.append("")
        if not perf_regress.empty:
            lines.append(f"### ⏱️ Performance Regressions (ratio < { (1 - E2E_PERFORMANCE_THRESHOLD) * 100:.0f}%)\n")
            cols = ['suite', 'dtype', 'mode', 'model', 'ind_tgt', 'eag_tgt', 'ind_bsl', 'eag_bsl', 'ind_ratio', 'eag_ratio', 'cmp_perf']
            available = [c for c in cols if c in perf_regress.columns]
            lines.extend(write_html_table(perf_regress, available, 'cmp_perf', fail_color="#f8d7da"))
            lines.append("")

    # ----- New Passes & Improvements -----
    acc_pass = details[details['cmp_acc'] == 'new_passed']
    perf_impr = details[details['cmp_perf'].isin(['new_improved', 'new_passed'])]
    if not acc_pass.empty or not perf_impr.empty:
        lines.append("## ✅ New Passes & Improvements\n")
        if not acc_pass.empty:
            lines.append("### 🧪 Accuracy New Passes\n")
            cols = ['suite', 'dtype', 'mode', 'model', 'bs_acc_tgt', 'acc_tgt', 'bs_acc_bsl', 'acc_bsl', 'cmp_acc']
            available = [c for c in cols if c in acc_pass.columns]
            lines.extend(write_html_table(acc_pass, available, 'cmp_acc', pass_color="#d4edda"))
            lines.append("")
        if not perf_impr.empty:
            lines.append(f"### ⏱️ Performance Improvements (ratio > { (1 + E2E_PERFORMANCE_THRESHOLD) * 100:.0f}%)\n")
            cols = ['suite', 'dtype', 'mode', 'model', 'ind_tgt', 'eag_tgt', 'ind_bsl', 'eag_bsl', 'ind_ratio', 'eag_ratio', 'cmp_perf']
            available = [c for c in cols if c in perf_impr.columns]
            lines.extend(write_html_table(perf_impr, available, 'cmp_perf', pass_color="#d4edda"))
            lines.append("")

    # ----- Suggestions -----
    suggestions = []
    if not acc_fail.empty:
        suggestions.append("❌ Investigate the new accuracy failures.")
    if not perf_regress.empty:
        suggestions.append("📉 Review performance regressions; they may indicate a real slowdown.")
    if acc_pass.empty and perf_impr.empty:
        suggestions.append("ℹ️ No new passes or improvements detected.")
    else:
        if not acc_pass.empty:
            suggestions.append("✅ New accuracy passes are good; ensure they are not due to test changes.")
        if not perf_impr.empty:
            suggestions.append("🚀 Performance improvements are encouraging; verify they are consistent.")
    if suggestions:
        lines.append("## 💡 Suggestions\n")
        for s in suggestions:
            lines.append(f"- {s}")
        lines.append("")
    else:
        lines.append("- All metrics are stable. No action required.\n")

    return "\n".join(lines)

# ---------- E2E main ----------
def e2e_main(args):
    target_files = e2e_find_result_files(args.target_dir)
    baseline_files = e2e_find_result_files(args.baseline_dir)
    print(f"Found {len(target_files)} CSV files in target directory.")
    print(f"Found {len(baseline_files)} CSV files in baseline directory.")

    target_acc = e2e_load_results(target_files, "accuracy")
    target_perf = e2e_load_results(target_files, "performance")
    baseline_acc = e2e_load_results(baseline_files, "accuracy")
    baseline_perf = e2e_load_results(baseline_files, "performance")

    print(f"Target accuracy records: {len(target_acc)}")
    print(f"Target performance records: {len(target_perf)}")
    print(f"Baseline accuracy records: {len(baseline_acc)}")
    print(f"Baseline performance records: {len(baseline_perf)}")

    acc_merged = e2e_merge_accuracy(target_acc, baseline_acc)
    perf_merged = e2e_merge_performance(target_perf, baseline_perf)

    combined_summary = e2e_generate_all_summaries(acc_merged, perf_merged)
    details = e2e_combine_results(acc_merged, perf_merged)

    combined_summary.rename(columns={k: v for k, v in E2E_COLUMN_RENAME_MAP.items() if k in combined_summary.columns}, inplace=True)
    details.rename(columns={k: v for k, v in E2E_COLUMN_RENAME_MAP.items() if k in details.columns}, inplace=True)

    if args.markdown:
        md = e2e_generate_markdown_string(combined_summary, details)
        with open(args.markdown, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"Markdown summary written to {args.markdown}")

    out_base, out_ext = os.path.splitext(args.output)
    if out_ext == '.xlsx':
        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            if not combined_summary.empty:
                combined_summary.to_excel(writer, sheet_name="Summary", index=False)
            else:
                pd.DataFrame({"Info": ["No summary data available"]}).to_excel(writer, sheet_name="Summary", index=False)
            if not details.empty:
                details.to_excel(writer, sheet_name="Details", index=False)
            else:
                pd.DataFrame({"Info": ["No detailed data available"]}).to_excel(writer, sheet_name="Details", index=False)
        print(f"Excel written to {args.output} (sheets: Summary, Details)")
    else:  # .csv
        summary_file = out_base + "_summary.csv"
        details_file = out_base + "_details.csv"
        if not combined_summary.empty:
            combined_summary.to_csv(summary_file, index=False, na_rep='')
            print(f"Summary written to {summary_file}")
        else:
            print("No summary data to write.")
        if not details.empty:
            details.to_csv(details_file, index=False, na_rep='')
            print(f"Details written to {details_file}")
        else:
            print("No detailed data to write.")

# ----------------------------------------------------------------------
# UT mode (unit test JUnit XML comparison)
# ----------------------------------------------------------------------

# ---------- UT enums ----------
class TestStatus(Enum):
    PASSED = "passed"
    XFAIL = "xfail"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, status_str: str) -> 'TestStatus':
        if not status_str or pd.isna(status_str):
            return cls.UNKNOWN
        status_str = str(status_str).lower().strip()
        status_mapping = {
            "pass": cls.PASSED,
            "success": cls.PASSED,
            "xfail": cls.XFAIL,
            "fail": cls.FAILED,
            "error": cls.ERROR,
            "skip": cls.SKIPPED,
        }
        for key, status in status_mapping.items():
            if key in status_str:
                return status
        return cls.UNKNOWN

    @property
    def priority(self) -> int:
        priorities = {
            self.PASSED: 5,
            self.XFAIL: 4,
            self.FAILED: 3,
            self.ERROR: 2,
            self.SKIPPED: 1,
            self.UNKNOWN: 0,
        }
        return priorities[self]

    @property
    def emoji(self) -> str:
        emojis = {
            self.PASSED: "✅",
            self.XFAIL: "⚠️",
            self.FAILED: "❌",
            self.ERROR: "💥",
            self.SKIPPED: "⏭️",
            self.UNKNOWN: "❓",
        }
        return emojis[self]

    @property
    def color(self) -> str:
        colors = {
            self.PASSED: "green",
            self.XFAIL: "yellow",
            self.FAILED: "red",
            self.ERROR: "red",
            self.SKIPPED: "gray",
            self.UNKNOWN: "gray",
        }
        return colors[self]

class TestDevice(Enum):
    BASELINE = "baseline"
    TARGET = "target"
    UNKNOWN = "unknown"

    @classmethod
    def from_test_type(cls, test_type: str) -> 'TestDevice':
        test_type_lower = test_type.lower()
        if "baseline" in test_type_lower:
            return cls.BASELINE
        elif "target" in test_type_lower:
            return cls.TARGET
        return cls.UNKNOWN

    @property
    def display_name(self) -> str:
        return self.value.capitalize()

# ---------- UT data classes ----------
@dataclasses.dataclass(frozen=True)
class TestCase:
    uniqname: str
    testfile: str
    classname: str
    name: str
    device: TestDevice
    testtype: str
    status: TestStatus
    time: float
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uniqname": self.uniqname,
            "testfile": self.testfile,
            "classname": self.classname,
            "name": self.name,
            "device": self.device.value,
            "testtype": self.testtype,
            "status": self.status.value,
            "time": float(self.time),
            "message": self.message,
        }

@dataclasses.dataclass
class Comparison:
    uniqname: str
    testfile_baseline: str = ""
    classname_baseline: str = ""
    name_baseline: str = ""
    testfile_target: str = ""
    classname_target: str = ""
    name_target: str = ""
    device_baseline: str = ""
    testtype_baseline: str = ""
    status_baseline: str = ""
    time_baseline: float = 0.0
    message_baseline: str = ""
    device_target: str = ""
    testtype_target: str = ""
    status_target: str = ""
    time_target: float = 0.0
    message_target: str = ""
    issue_ids: str = ""
    issue_labels: str = ""
    issue_statuses: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

# ---------- UT GitHub issue tracker ----------
class FilePatternMatcher:
    _CLASSNAME_PATTERN = re.compile(r".*\.")
    _CASENAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")
    _TESTFILE_PATTERN = re.compile(r".*torch-xpu-ops\.test\.")
    _TESTFILE_PATTERN_CPP = re.compile(r".*/test/xpu/")
    _NORMALIZE_PATTERN = re.compile(r".*\.\./test/")
    _GPU_PATTERN = re.compile(r"(?:xpu|cuda)", re.IGNORECASE)

    TEST_TYPE_PATTERNS = {
        "xpu-target": [r"/target/"],
        "xpu-baseline": [r"/baseline/"],
    }

    FILE_REPLACEMENTS = [
        ("test/test/", "test/"),
        ("test_c10d_xccl.py", "test_c10d_nccl.py"),
        ("test_c10d_ops_xccl.py", "test_c10d_ops_nccl.py"),
    ]

    def __init__(self):
        self._compiled_test_type_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        return {
            test_type: [re.compile(pattern) for pattern in patterns]
            for test_type, patterns in self.TEST_TYPE_PATTERNS.items()
        }

    @lru_cache(maxsize=1024)
    def determine_test_type(self, xml_file: Path) -> str:
        xml_file_str = str(xml_file)
        for test_type, patterns in self._compiled_test_type_patterns.items():
            if any(pattern.search(xml_file_str) for pattern in patterns):
                return test_type
        return "unknown"

    @lru_cache(maxsize=1024)
    def normalize_filepath(self, filepath: str) -> str:
        if not filepath:
            return "unknown_file.py"
        normalized = filepath
        if self._NORMALIZE_PATTERN.search(normalized):
            normalized = self._NORMALIZE_PATTERN.sub("test/", normalized)
        for old, new in self.FILE_REPLACEMENTS:
            if old in normalized:
                normalized = normalized.replace(old, new)
        normalized = normalized.replace("_xpu_xpu.py", ".py").replace("_xpu.py", ".py")
        normalized = re.sub(r'.*/jenkins/workspace/', '', normalized, flags=re.IGNORECASE)
        return normalized or "unknown_file.py"

    @lru_cache(maxsize=1024)
    def extract_testfile(self, classname: str, filename: str, xml_file: Path) -> str:
        if filename:
            if filename.endswith(".cpp"):
                testfile = self._TESTFILE_PATTERN_CPP.sub("test/", filename)
            elif filename.endswith(".py"):
                testfile = f"test/{filename}"
            else:
                testfile = filename
        elif classname:
            testfile = self._TESTFILE_PATTERN.sub("test/", classname).replace(".", "/")
            if "/" in testfile:
                testfile = f"{testfile.rsplit('/', 1)[0]}.py"
            else:
                testfile = f"{testfile}.py"
        else:
            xml_file_str = str(xml_file)
            testfile = (
                re.sub(r".*op_ut_with_[a-zA-Z0-9]+\.", "test.", xml_file_str)
                .replace(".", "/")
                .replace("/py/xml", ".py")
                .replace("/xml", ".py")
            )
        return self.normalize_filepath(testfile)

    @lru_cache(maxsize=1024)
    def extract_classname(self, full_classname: str) -> str:
        if not full_classname:
            return "UnknownClass"
        return self._CLASSNAME_PATTERN.sub("", full_classname)

    def extract_casename(self, casename: str) -> str:
        if not casename:
            return "unknown_name"
        return self._CASENAME_PATTERN.sub("", casename) or "error_name"

    @lru_cache(maxsize=2048)
    def generate_uniqname(self, filename: str, classname: str, name: str) -> str:
        combined = f"{filename}{classname}{name}"
        return self._GPU_PATTERN.sub("cuda", combined)

class GitHubIssueTracker:
    CASES_PATTERN = re.compile(r'Cases:\s*\n(.*?)(?:\n\n|\Z)', re.DOTALL | re.IGNORECASE)
    TEST_CASE_SPLIT_PATTERN = re.compile(r'[\n]+')

    def __init__(self, repo: str = None, token: str = None, cache_path: str = None, pattern_matcher: Optional[FilePatternMatcher] = None):
        self.repo_name = repo or os.environ.get('GITHUB_REPOSITORY', '')
        self.token = token or os.environ.get('GITHUB_TOKEN', '')
        self.cache_path = Path(cache_path) if cache_path else None
        self.github = None
        self.repository = None
        self.issues_cache: Dict[int, Dict[str, Any]] = {}
        self.test_to_issues: Dict[str, List[Dict[str, Any]]] = {}
        self.pattern_matcher = pattern_matcher or FilePatternMatcher()
        if not GITHUB_AVAILABLE:
            logging.getLogger(__name__).error("PyGithub is not installed. GitHub integration disabled.")

    def load_cache(self) -> bool:
        if not self.cache_path or not self.cache_path.exists():
            return False
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.issues_cache = {int(k): v for k, v in data.get('issues_cache', {}).items()}
            self.test_to_issues = data.get('test_to_issues', {})
            logging.getLogger(__name__).info(f"Loaded {len(self.issues_cache)} issues from cache: {self.cache_path}")
            return True
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load cache from {self.cache_path}: {e}")
            return False

    def save_cache(self) -> bool:
        if not self.cache_path:
            return False
        try:
            data = {
                'issues_cache': {str(k): v for k, v in self.issues_cache.items()},
                'test_to_issues': self.test_to_issues
            }
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.getLogger(__name__).info(f"Saved {len(self.issues_cache)} issues to cache: {self.cache_path}")
            return True
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to save cache to {self.cache_path}: {e}")
            return False

    def fetch_issues(self, state: str = 'all', labels: List[str] = None, force_refresh: bool = False) -> bool:
        if not force_refresh and self.load_cache():
            return True
        if not self.repository:
            if not self._init_github():
                return False
        logging.getLogger(__name__).info(f"Fetching issues from {self.repo_name} (state={state})")
        try:
            github_state = 'all' if state == 'all' else state
            kwargs = {'state': github_state, 'direction': 'desc'}
            if labels:
                kwargs['labels'] = labels
            issues = self.repository.get_issues(**kwargs)
            self.issues_cache.clear()
            self.test_to_issues.clear()
            issue_count = 0
            for issue in issues:
                if issue.pull_request:
                    continue
                self._parse_issue(issue)
                issue_count += 1
            logging.getLogger(__name__).info(f"Fetched {issue_count} issues, found {len(self.test_to_issues)} test mappings")
            self.save_cache()
            return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Error fetching issues from GitHub: {e}")
            return False

    def _init_github(self) -> bool:
        try:
            if self.token:
                auth = Auth.Token(self.token)
                self.github = Github(auth=auth)
            else:
                self.github = Github()
            self.repository = self.github.get_repo(self.repo_name)
            logging.getLogger(__name__).info(f"Connected to GitHub repository: {self.repo_name}")
            return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to initialize GitHub client: {e}")
            return False

    def _parse_issue(self, issue: Issue) -> None:
        issue_id = issue.number
        issue_body = issue.body or ''
        issue_labels = [label.name for label in issue.labels]
        issue_state = issue.state
        issue_title = issue.title
        self.issues_cache[issue_id] = {
            'id': issue_id,
            'title': issue_title,
            'state': issue_state,
            'labels': issue_labels,
            'url': issue.html_url,
            'created_at': issue.created_at.isoformat() if issue.created_at else '',
            'updated_at': issue.updated_at.isoformat() if issue.updated_at else '',
        }
        test_cases = self._extract_test_cases(issue_body)
        for test_case in test_cases:
            parts = test_case.split(',')
            if len(parts) < 3:
                logging.getLogger(__name__).debug(f"Issue #{issue_id}: test case '{test_case}' has fewer than 3 comma-separated parts, skipping")
                continue
            class_name_raw = parts[1].strip()
            test_name_raw = parts[2].strip()
            test_file = self.pattern_matcher.extract_testfile(class_name_raw, None, None)
            test_class = self.pattern_matcher.extract_classname(class_name_raw)
            test_name = self.pattern_matcher.extract_casename(test_name_raw)
            uniqname = self.pattern_matcher.generate_uniqname(test_file, test_class, test_name)
            self.test_to_issues.setdefault(uniqname, []).append({
                'id': issue_id,
                'state': issue_state,
                'labels': issue_labels
            })

    def _extract_test_cases(self, body: str) -> List[str]:
        if not body:
            return []
        match = self.CASES_PATTERN.search(body)
        if not match:
            return []
        cases_text = match.group(1).strip()
        cases = self.TEST_CASE_SPLIT_PATTERN.split(cases_text)
        return [case.strip() for case in cases if case.strip()]

    def find_issues_for_test(self, test_uniqname: str) -> List[Dict[str, Any]]:
        return self.test_to_issues.get(test_uniqname, [])

    def enhance_comparison(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        if merged_df.empty:
            return merged_df
        enhanced_df = merged_df.copy()
        enhanced_df['issue_ids'] = ''
        enhanced_df['issue_labels'] = ''
        enhanced_df['issue_statuses'] = ''
        for idx, row in enhanced_df.iterrows():
            uniqname = row.get('uniqname', '')
            if not uniqname:
                continue
            issues = self.find_issues_for_test(uniqname)
            if issues:
                issue_ids = [str(issue['id']) for issue in issues]
                issue_statuses = [issue['state'] for issue in issues]
                all_labels = set()
                for issue in issues:
                    all_labels.update(issue['labels'])
                enhanced_df.at[idx, 'issue_ids'] = ','.join(issue_ids)
                enhanced_df.at[idx, 'issue_labels'] = ','.join(sorted(all_labels))
                enhanced_df.at[idx, 'issue_statuses'] = ','.join(issue_statuses)
        return enhanced_df

    def get_issue_summary_stats(self) -> Dict[str, Any]:
        stats = {
            'total_issues': len(self.issues_cache),
            'open_issues': 0,
            'closed_issues': 0,
            'issues_with_test_cases': len(set(
                issue['id'] for mappings in self.test_to_issues.values() for issue in mappings
            )),
            'unique_test_cases': len(self.test_to_issues),
            'labels': {},
        }
        for issue in self.issues_cache.values():
            if issue['state'] == 'open':
                stats['open_issues'] += 1
            else:
                stats['closed_issues'] += 1
            for label in issue['labels']:
                stats['labels'][label] = stats['labels'].get(label, 0) + 1
        return stats

# ---------- UT test details extractor ----------
class TestDetailsExtractor:
    def __init__(self, pattern_matcher: Optional[FilePatternMatcher] = None):
        self.pattern_matcher = pattern_matcher or FilePatternMatcher()
        self.test_cases: List[TestCase] = []
        self.stats = {
            "files_processed": 0,
            "test_cases_found": 0,
            "empty_files": 0,
            "failed_files": 0,
        }

    def _determine_test_status(self, testcase: ET.Element) -> Tuple[TestStatus, str]:
        failure = testcase.find("failure")
        if failure is not None:
            message = failure.get("message", "")
            if "pytest.xfail" in message:
                return TestStatus.XFAIL, message
            return TestStatus.FAILED, message
        skipped = testcase.find("skipped")
        if skipped is not None:
            message = skipped.get("message", "")
            skip_type = skipped.get("type", "")
            if "pytest.xfail" in skip_type or "pytest.xfail" in message:
                return TestStatus.XFAIL, message
            return TestStatus.SKIPPED, message
        error = testcase.find("error")
        if error is not None:
            return TestStatus.ERROR, error.get("message", "")
        return TestStatus.PASSED, ""

    def _parse_testcase(self, testcase: ET.Element, xml_file: Path) -> Optional[TestCase]:
        try:
            classname = testcase.get("classname", "")
            filename = testcase.get("file", "")
            name = testcase.get("name", "")
            time_str = testcase.get("time", "0")
            test_type = self.pattern_matcher.determine_test_type(xml_file)
            device = TestDevice.from_test_type(test_type)
            if device == TestDevice.UNKNOWN:
                return None
            simplified_classname = self.pattern_matcher.extract_classname(classname)
            simplified_casename = self.pattern_matcher.extract_casename(name)
            testfile = self.pattern_matcher.extract_testfile(classname, filename, xml_file)
            uniqname = self.pattern_matcher.generate_uniqname(
                testfile, simplified_classname, simplified_casename
            )
            status, message = self._determine_test_status(testcase)
            try:
                time_val = float(time_str)
            except (ValueError, TypeError):
                time_val = 0.0
            return TestCase(
                uniqname=uniqname,
                testfile=testfile,
                classname=simplified_classname,
                name=simplified_casename,
                device=device,
                testtype=test_type,
                status=status,
                time=time_val,
                message=message,
            )
        except Exception as e:
            logging.getLogger(__name__).debug(f"Error parsing test case in {xml_file}: {e}")
            return None

    def process_xml(self, xml_file: Path) -> List[TestCase]:
        try:
            test_cases = []
            for event, elem in ET.iterparse(xml_file, events=('end',)):
                if elem.tag == 'testcase':
                    test_case = self._parse_testcase(elem, xml_file)
                    if test_case:
                        test_cases.append(test_case)
                    elem.clear()
            return test_cases
        except Exception as e:
            logging.getLogger(__name__).error(f"Error processing {xml_file}: {e}")
            self.stats["failed_files"] += 1
            return []

    def find_xml_files(self, input_paths: List[str]) -> List[Path]:
        xml_files: Set[Path] = set()
        for input_path in input_paths:
            path = Path(input_path).expanduser().resolve()
            if path.is_file() and path.suffix.lower() == ".xml":
                xml_files.add(path)
            elif path.is_dir():
                xml_files.update(path.rglob("*.xml"))
            else:
                for file_path in glob.glob(str(path), recursive=True):
                    file_path = Path(file_path)
                    if file_path.is_file() and file_path.suffix.lower() == ".xml":
                        xml_files.add(file_path.resolve())
        return sorted(xml_files)

    def process(self, input_paths: List[str], max_workers: int = None) -> bool:
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 2)
        xml_files = self.find_xml_files(input_paths)
        if not xml_files:
            logging.getLogger(__name__).error("No XML files found")
            return False
        logging.getLogger(__name__).info(f"Found {len(xml_files)} XML files, using {max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_xml, xml_file): xml_file
                for xml_file in xml_files
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                xml_file = future_to_file[future]
                completed += 1
                if completed % 10 == 0 or completed == len(xml_files):
                    logging.getLogger(__name__).info(f"Processed {completed}/{len(xml_files)} files")
                try:
                    test_cases = future.result()
                    if test_cases:
                        self.test_cases.extend(test_cases)
                        self.stats["test_cases_found"] += len(test_cases)
                    else:
                        self.stats["empty_files"] += 1
                except Exception as e:
                    logging.getLogger(__name__).error(f"Error processing {xml_file}: {e}")
                    self.stats["failed_files"] += 1
                finally:
                    self.stats["files_processed"] += 1
        return len(self.test_cases) > 0

# ---------- UT result analyzer ----------
class ResultAnalyzer:
    def __init__(self, test_cases: List[TestCase]):
        self.test_cases = test_cases
        self.df = self._create_dataframe()
        self.issue_tracker: Optional[GitHubIssueTracker] = None

    def set_issue_tracker(self, issue_tracker: GitHubIssueTracker):
        self.issue_tracker = issue_tracker

    def _create_dataframe(self) -> pd.DataFrame:
        if not self.test_cases:
            return pd.DataFrame()
        data = [tc.to_dict() for tc in self.test_cases]
        return pd.DataFrame(data)

    def deduplicate_by_priority(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()
        df = self.df.copy()
        df["_priority"] = df["status"].apply(
            lambda x: TestStatus.from_string(x).priority
        )
        df_sorted = df.sort_values("_priority", ascending=False)
        result = df_sorted.drop_duplicates(subset=["device", "uniqname"], keep="first")
        return result.drop(columns=["_priority"]).reset_index(drop=True)

    def split_by_device(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df is None:
            df = self.df
        if df.empty or "device" not in df.columns:
            return pd.DataFrame(), pd.DataFrame()
        baseline_mask = df["device"] == "baseline"
        target_mask = df["device"] == "target"
        return df[baseline_mask].copy(), df[target_mask].copy()

    def merge_results(self, baseline_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        if baseline_df.empty and target_df.empty:
            return pd.DataFrame()
        baseline_clean = baseline_df.add_suffix("_baseline").rename(columns={"uniqname_baseline": "uniqname"})
        target_clean = target_df.add_suffix("_target").rename(columns={"uniqname_target": "uniqname"})
        merged = pd.merge(
            baseline_clean,
            target_clean,
            on="uniqname",
            how="outer",
            suffixes=("", "_dup"),
        ).fillna("")
        columns = [
            "uniqname",
            "testfile_baseline", "classname_baseline", "name_baseline",
            "testfile_target", "classname_target", "name_target",
            "device_baseline", "testtype_baseline", "status_baseline", "time_baseline", "message_baseline",
            "device_target", "testtype_target", "status_target", "time_target", "message_target",
        ]
        existing_cols = [col for col in columns if col in merged.columns]
        merged_df = merged[existing_cols]
        if self.issue_tracker:
            logging.getLogger(__name__).info("Enhancing comparison with GitHub issue information")
            merged_df = self.issue_tracker.enhance_comparison(merged_df)
        return merged_df

    def find_target_changes(self, merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if merged_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        merged_df = merged_df.copy()
        baseline_passed = merged_df["status_baseline"].isin(["passed", "xfail"])
        target_not_passed = ~merged_df["status_target"].isin(["passed", "xfail"])
        new_failures = merged_df[baseline_passed & target_not_passed].copy()
        baseline_not_passed = ~merged_df["status_baseline"].isin(["passed", "xfail"])
        target_passed = merged_df["status_target"].isin(["passed", "xfail"])
        new_passes = merged_df[baseline_not_passed & target_passed].copy()
        if not new_failures.empty:
            new_failures["change_type"] = "failure"
            new_failures["reason"] = np.select(
                [
                    new_failures["status_target"].isin(["skipped"]),
                    new_failures["status_target"].isin(["failed"]),
                    new_failures["status_target"].isin(["error"]),
                ],
                [
                    "Skipped on Target",
                    "Failed on Target",
                    "Error on Target",
                ],
                default="Unknown issue"
            )
        if not new_passes.empty:
            new_passes["change_type"] = "pass"
            new_passes["reason"] = np.select(
                [
                    new_passes["status_baseline"].isin(["skipped"]),
                    new_passes["status_baseline"].isin(["failed"]),
                    new_passes["status_baseline"].isin(["error"]),
                ],
                [
                    "Was skipped on Baseline",
                    "Was failing on Baseline",
                    "Was error on Baseline",
                ],
                default="Now passing on Target"
            )
        return new_failures, new_passes

    def generate_file_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        baseline_df, target_df = self.split_by_device(df)
        if baseline_df.empty or target_df.empty:
            return pd.DataFrame()
        all_testfiles = sorted(set(baseline_df["testfile"].unique()) | set(target_df["testfile"].unique()))
        file_summaries = []
        for testfile in all_testfiles:
            baseline_file_df = baseline_df[baseline_df["testfile"] == testfile]
            target_file_df = target_df[target_df["testfile"] == testfile]
            summary = {
                "Test File": testfile,
                "Baseline Total": len(baseline_file_df),
                "Baseline Passed": len(baseline_file_df[baseline_file_df["status"].isin(["passed", "xfail"])]),
                "Baseline Failed": len(baseline_file_df[baseline_file_df["status"] == "failed"]),
                "Baseline Error": len(baseline_file_df[baseline_file_df["status"] == "error"]),
                "Baseline Skipped": len(baseline_file_df[baseline_file_df["status"] == "skipped"]),
                "Target Total": len(target_file_df),
                "Target Passed": len(target_file_df[target_file_df["status"].isin(["passed", "xfail"])]),
                "Target Failed": len(target_file_df[target_file_df["status"] == "failed"]),
                "Target Error": len(target_file_df[target_file_df["status"] == "error"]),
                "Target Skipped": len(target_file_df[target_file_df["status"] == "skipped"]),
            }
            baseline_total = summary["Baseline Total"]
            if baseline_total > 0:
                baseline_passed_count = summary["Baseline Passed"]
                summary["Baseline Pass Rate"] = f"{(baseline_passed_count / baseline_total * 100):.2f}%"
            else:
                summary["Baseline Pass Rate"] = "N/A"
            target_total = summary["Target Total"]
            if target_total > 0:
                target_passed_count = summary["Target Passed"]
                summary["Target Pass Rate"] = f"{(target_passed_count / target_total * 100):.2f}%"
            else:
                summary["Target Pass Rate"] = "N/A"
            if baseline_total > 0 and target_total > 0:
                baseline_passed_count = summary["Baseline Passed"]
                target_passed_count = summary["Target Passed"]
                baseline_passed_pct = baseline_passed_count / baseline_total * 100
                target_passed_pct = target_passed_count / target_total * 100
                summary["Pass Rate Delta"] = f"{(target_passed_pct - baseline_passed_pct):+.2f}%"
            else:
                summary["Pass Rate Delta"] = "N/A"
            file_summaries.append(summary)
        return pd.DataFrame(file_summaries)

    def generate_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        stats = []
        for device_value in ["baseline", "target"]:
            device_df = df[df["device"] == device_value]
            if device_df.empty:
                continue
            total = len(device_df)
            failed = len(device_df[device_df["status"] == "failed"])
            error = len(device_df[device_df["status"] == "error"])
            passed_count = total - failed - error
            pass_rate = (passed_count / total * 100) if total > 0 else 0
            device_name = "Baseline" if device_value == "baseline" else "Target"
            stats.append({
                "Device": device_name,
                "Total": total,
                "Passed": passed_count,
                "Failed": failed,
                "Error": error,
                "Skipped": len(device_df[device_df["status"] == "skipped"]),
                "XFAIL": len(device_df[device_df["status"] == "xfail"]),
                "Pass Rate": f"{pass_rate:.2f}%",
            })
        return pd.DataFrame(stats)

    def generate_issue_summary_section(self) -> str:
        if not self.issue_tracker:
            return ""
        stats = self.issue_tracker.get_issue_summary_stats()
        md = []
        md.append("## 🏷️ GitHub Issues Summary\n")
        md.append(f"- **Total Issues:** {stats['total_issues']}")
        md.append(f"  - 🔓 Open: {stats['open_issues']}")
        md.append(f"  - 🔒 Closed: {stats['closed_issues']}")
        md.append(f"- **Issues with Test Cases:** {stats['issues_with_test_cases']}")
        md.append(f"- **Unique Test Cases Tracked:** {stats['unique_test_cases']}")
        if stats['labels']:
            top_labels = sorted(stats['labels'].items(), key=lambda x: x[1], reverse=True)[:5]
            md.append("\n**Top Labels:**")
            for label, count in top_labels:
                md.append(f"  - `{label}`: {count} issues")
        md.append("")
        return "\n".join(md)

    def generate_markdown_summary(self, df: pd.DataFrame, new_failures_df: pd.DataFrame = None, new_passes_df: pd.DataFrame = None) -> str:
        """Return markdown string; empty if no data."""
        if df.empty:
            return ""
        stats_df = self.generate_summary_stats(df)
        file_summary_df = self.generate_file_summary(df)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append("# Test Comparison Report\n")
        lines.append(f"**Generated:** {timestamp}\n")
        issue_summary = self.generate_issue_summary_section()
        if issue_summary:
            lines.append(issue_summary)
        if not stats_df.empty:
            lines.append("## 📊 Overall Summary\n")
            lines.append("| Device | Total | ✅ Passed | ❌ Failed | 💥 Error | ⏭️ Skipped | ⚠️ XFAIL | 📈 Pass Rate |")
            lines.append("|--------|-------|----------|----------|---------|-----------|---------|--------------|")
            for _, row in stats_df.iterrows():
                lines.append(
                    f"| {row['Device']} | {row['Total']} | {row['Passed']} | "
                    f"{row['Failed']} | {row['Error']} | {row['Skipped']} | "
                    f"{row['XFAIL']} | {row['Pass Rate']} |"
                )
            lines.append("")
        if len(stats_df) == 2:
            baseline_row = stats_df[stats_df['Device'] == 'Baseline'].iloc[0]
            target_row = stats_df[stats_df['Device'] == 'Target'].iloc[0]
            total_delta = target_row['Total'] - baseline_row['Total']
            pass_rate_delta = float(target_row['Pass Rate'].rstrip('%')) - float(baseline_row['Pass Rate'].rstrip('%'))
            delta_emoji = "✅" if pass_rate_delta >= 0 else "❌"
            lines.append("### 🔄 Comparison Metrics\n")
            lines.append(f"- **Test Count Delta:** {total_delta:+.0f} tests")
            lines.append(f"- **Pass Rate Delta:** {delta_emoji} {pass_rate_delta:+.2f}%\n")
        total_new_failures = len(new_failures_df) if new_failures_df is not None else 0
        total_new_passes = len(new_passes_df) if new_passes_df is not None else 0
        lines.append("### 📊 Change Summary\n")
        lines.append(f"- **New Failures:** {total_new_failures} tests")
        lines.append(f"- **New Passes:** {total_new_passes} tests")
        lines.append(f"- **Net Change:** {total_new_passes - total_new_failures:+.0f} tests\n")

        if new_failures_df is not None and not new_failures_df.empty:
            lines.append("## 🚨 New Failures on Target\n")
            lines.append(f"Found **{len(new_failures_df)}** tests that passed on Baseline but failed on Target:\n")
            for reason in new_failures_df['reason'].unique():
                reason_issues = new_failures_df[new_failures_df['reason'] == reason]
                reason_emoji = "❌" if "Failed" in reason else "💥" if "Error" in reason else "⏭️"
                lines.append(f"### {reason_emoji} {reason} ({len(reason_issues)})\n")
                lines.append("| Test File | Test Name | Status | Issue IDs | Message |")
                lines.append("|-----------|-----------|--------|-----------|---------|")
                for _, issue in reason_issues.head(10).iterrows():
                    test_file = issue.get('testfile_target', issue.get('testfile_baseline', 'Unknown'))
                    test_name = issue.get('name_target', issue.get('name_baseline', 'Unknown'))
                    status = issue['status_target']
                    status_emoji = TestStatus.from_string(status).emoji
                    issue_ids = issue.get('issue_ids', '')
                    issue_display = ""
                    if issue_ids and self.issue_tracker and self.issue_tracker.repo_name:
                        ids = issue_ids.split(',')
                        issue_links = [f"[#{id}](https://github.com/{self.issue_tracker.repo_name}/issues/{id})" for id in ids]
                        issue_display = ', '.join(issue_links)
                    elif issue_ids:
                        issue_display = issue_ids
                    message = issue.get('message_target', '')
                    if len(message) > 100:
                        message = message[:97] + "..."
                    message = message.replace('\n', ' ').replace('|', '\\|')
                    lines.append(f"| `{test_file}` | `{test_name}` | {status_emoji} {status} | {issue_display} | {message} |")
                if len(reason_issues) > 10:
                    lines.append(f"| ... | ... | ... | ... | *{len(reason_issues) - 10} more issues* |")
                lines.append("")

        if new_passes_df is not None and not new_passes_df.empty:
            lines.append("## ✨ New Passes on Target\n")
            lines.append(f"Found **{len(new_passes_df)}** tests that now pass on Target (were failing/skipped on Baseline):\n")
            for reason in new_passes_df['reason'].unique():
                reason_passes = new_passes_df[new_passes_df['reason'] == reason]
                reason_emoji = "✅"
                lines.append(f"### {reason_emoji} {reason} ({len(reason_passes)})\n")
                lines.append("| Test File | Test Name | Baseline Status | Issue IDs |")
                lines.append("|-----------|-----------|-----------------|-----------|")
                for _, issue in reason_passes.head(10).iterrows():
                    test_file = issue.get('testfile_target', issue.get('testfile_baseline', 'Unknown'))
                    test_name = issue.get('name_target', issue.get('name_baseline', 'Unknown'))
                    baseline_status = issue['status_baseline']
                    status_emoji = TestStatus.from_string(baseline_status).emoji
                    issue_ids = issue.get('issue_ids', '')
                    issue_display = ""
                    if issue_ids and self.issue_tracker and self.issue_tracker.repo_name:
                        ids = issue_ids.split(',')
                        issue_links = [f"[#{id}](https://github.com/{self.issue_tracker.repo_name}/issues/{id})" for id in ids]
                        issue_display = ', '.join(issue_links)
                    elif issue_ids:
                        issue_display = issue_ids
                    lines.append(f"| `{test_file}` | `{test_name}` | {status_emoji} {baseline_status} | {issue_display} |")
                if len(reason_passes) > 10:
                    lines.append(f"| ... | ... | ... | *{len(reason_passes) - 10} more passes* |")
                lines.append("")

        if not file_summary_df.empty:
            lines.append("## 📁 File-Level Summary\n")
            file_summary_sorted = file_summary_df.copy()
            file_summary_sorted['Baseline Failures'] = (
                file_summary_sorted['Baseline Failed'].astype(int) +
                file_summary_sorted['Baseline Error'].astype(int)
            )
            file_summary_sorted = file_summary_sorted.sort_values(
                by=['Baseline Failures', 'Baseline Total'],
                ascending=[False, False]
            )
            file_summary_sorted['Delta Numeric'] = file_summary_sorted['Pass Rate Delta'].apply(
                lambda x: float(x.rstrip('%')) if x != 'N/A' and x != 'N/A%' else 0
            )
            lines.append("| Test File | Baseline Stats | Target Stats | Delta | Details |")
            lines.append("|-----------|---------------|--------------|-------|---------|")
            for _, row in file_summary_sorted.head(10).iterrows():
                test_file = row['Test File']
                baseline_rate = row['Baseline Pass Rate']
                target_rate = row['Target Pass Rate']
                delta = row['Pass Rate Delta']
                delta_emoji = ""
                if delta != 'N/A':
                    delta_val = float(delta.rstrip('%'))
                    if delta_val < -5:
                        delta_emoji = "🔻"
                    elif delta_val > 5:
                        delta_emoji = "🔺"
                baseline_failures = int(row['Baseline Failed']) + int(row['Baseline Error'])
                baseline_stats = f"✅:{row['Baseline Passed']}"
                if baseline_failures > 0:
                    baseline_stats += f" ❌:{row['Baseline Failed']} 💥:{row['Baseline Error']}"
                baseline_stats += f" ⏭️:{row['Baseline Skipped']}"
                target_failures = int(row['Target Failed']) + int(row['Target Error'])
                target_stats = f"✅:{row['Target Passed']}"
                if target_failures > 0:
                    target_stats += f" ❌:{row['Target Failed']} 💥:{row['Target Error']}"
                target_stats += f" ⏭️:{row['Target Skipped']}"
                details = f"Total: {row['Baseline Total']} tests"
                lines.append(
                    f"| `{test_file}` | {baseline_stats} | {target_stats} | {delta_emoji} {delta} | {details} |"
                )
            if len(file_summary_df) > 10:
                lines.append(f"| ... | ... | ... | ... | *{len(file_summary_df) - 10} more files* |")
            lines.append("")
            lines.append("> 📝 *Files are sorted by Baseline Failures (Failed + Error) descending, then by total test count.*\n")
        else:
            lines.append("No file-level summary available.\n")

        if new_failures_df is not None and not new_failures_df.empty:
            target_failures = file_summary_df.nlargest(5, 'Target Failed')[['Test File', 'Target Failed', 'Target Error']]
            if not target_failures.empty and (target_failures['Target Failed'].sum() > 0 or target_failures['Target Error'].sum() > 0):
                lines.append("## 🔥 Top Failures by File\n")
                lines.append("| Test File | Failed | Error |")
                lines.append("|-----------|--------|-------|")
                for _, row in target_failures.iterrows():
                    if row['Target Failed'] > 0 or row['Target Error'] > 0:
                        lines.append(f"| `{row['Test File']}` | {row['Target Failed']} | {row['Target Error']} |")
                lines.append("")
        if new_passes_df is not None and not new_passes_df.empty:
            new_passes_by_file = new_passes_df.groupby('testfile_target').size().reset_index(name='new_passes_count')
            new_passes_by_file = new_passes_by_file.sort_values('new_passes_count', ascending=False).head(5)
            if not new_passes_by_file.empty:
                lines.append("## 📈 Top Improvements by File\n")
                lines.append("| Test File | New Passes |")
                lines.append("|-----------|------------|")
                for _, row in new_passes_by_file.iterrows():
                    lines.append(f"| `{row['testfile_target']}` | {row['new_passes_count']} |")
                lines.append("")

        # Recommendations
        if new_failures_df is not None and not new_failures_df.empty:
            lines.append("## 💡 Recommendations\n")
            lines.append("Based on the analysis, here are some recommendations:\n")
            lines.append(f"1. **🔥 Focus on new failures first** - The {len(new_failures_df)} new failures should be investigated urgently")
            lines.append("2. **Check error messages** - Review the error messages for patterns in the new failures")
            lines.append("3. **Verify test environment** - Ensure Target environment is properly configured")
            lines.append("4. **Review recent changes** - Changes between Baseline and Target may have introduced these issues")
            if not file_summary_df.empty:
                high_failure_files = file_summary_df[
                    (file_summary_df['Baseline Failed'].astype(int) + file_summary_df['Baseline Error'].astype(int)) > 10
                ]
                if not high_failure_files.empty:
                    lines.append(f"5. **Address baseline failures** - {len(high_failure_files)} files have >10 failures in baseline")
            if new_passes_df is not None and not new_passes_df.empty:
                lines.append(f"6. **✨ Celebrate improvements** - {len(new_passes_df)} tests are now passing on Target!")
            lines.append("")
        elif new_passes_df is not None and not new_passes_df.empty:
            lines.append("## 💡 Recommendations\n")
            lines.append("✅ All tests are passing! No immediate action required.\n")
            lines.append(f"✨ **Note:** {len(new_passes_df)} tests that were previously failing are now passing on Target!\n")
        else:
            lines.append("## 💡 Recommendations\n")
            lines.append("✅ All metrics are stable. No action required.\n")

        lines.append("---\n")
        lines.append("*This report was automatically generated by the Test Comparison Tool.*")
        lines.append("*For more details, check the attached Excel/CSV files.*")
        return "\n".join(lines)

# ---------- UT report exporter ----------
class ReportExporter:
    def __init__(self, markdown_output: Optional[Path] = None):
        self.markdown_output = markdown_output

    def export_excel(self, analyzer: ResultAnalyzer, output_path: Path) -> None:
        unique_df = analyzer.deduplicate_by_priority()
        if unique_df.empty:
            logging.getLogger(__name__).warning("No data to export")
            return
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            baseline_df, target_df = analyzer.split_by_device(unique_df)
            if not baseline_df.empty and not target_df.empty:
                merged_df = analyzer.merge_results(baseline_df, target_df)
                merged_df.to_excel(writer, sheet_name="Comparison", index=False)
                new_failures_df, new_passes_df = analyzer.find_target_changes(merged_df)
                if not new_failures_df.empty:
                    new_failures_df.to_excel(writer, sheet_name="New failures", index=False)
                if not new_passes_df.empty:
                    new_passes_df.to_excel(writer, sheet_name="New passes", index=False)
            file_summary_df = analyzer.generate_file_summary(unique_df)
            if not file_summary_df.empty:
                file_summary_df.to_excel(writer, sheet_name="Files summary", index=False)
            stats_df = analyzer.generate_summary_stats(unique_df)
            if not stats_df.empty:
                stats_df.to_excel(writer, sheet_name="Summary", index=False)
            case_issue_df = self._generate_case_issue_df(analyzer)
            if not case_issue_df.empty:
                case_issue_df.to_excel(writer, sheet_name="Case to issue", index=False)
                logging.getLogger(__name__).info("Exported case-to-issue mapping to sheet 'Case to issue'")
        logging.getLogger(__name__).info(f"Exported comparison results to {output_path}")

    def export_csv(self, analyzer: ResultAnalyzer, output_path: Path) -> None:
        unique_df = analyzer.deduplicate_by_priority()
        if unique_df.empty:
            logging.getLogger(__name__).warning("No data to export")
            return
        base_path = output_path.parent / output_path.stem
        baseline_df, target_df = analyzer.split_by_device(unique_df)
        if not baseline_df.empty and not target_df.empty:
            merged_df = analyzer.merge_results(baseline_df, target_df)
            merged_df.to_csv(f"{base_path}_comparison.csv", index=False)
            new_failures_df, new_passes_df = analyzer.find_target_changes(merged_df)
            if not new_failures_df.empty:
                new_failures_df.to_csv(f"{base_path}_new_failures.csv", index=False)
            if not new_passes_df.empty:
                new_passes_df.to_csv(f"{base_path}_new_passes.csv", index=False)
        file_summary_df = analyzer.generate_file_summary(unique_df)
        if not file_summary_df.empty:
            file_summary_df.to_csv(f"{base_path}_files_summary.csv", index=False)
        stats_df = analyzer.generate_summary_stats(unique_df)
        if not stats_df.empty:
            stats_df.to_csv(f"{base_path}_summary.csv", index=False)
        case_issue_df = self._generate_case_issue_df(analyzer)
        if not case_issue_df.empty:
            case_issue_df.to_csv(f"{base_path}_case_to_issue.csv", index=False)
            logging.getLogger(__name__).info(f"Exported case-to-issue mapping to {base_path}_case_to_issue.csv")
        logging.getLogger(__name__).info(f"Exported comparison results to {output_path}")

    def export_markdown(self, analyzer: ResultAnalyzer, output_path: Path) -> None:
        unique_df = analyzer.deduplicate_by_priority()
        if unique_df.empty:
            logging.getLogger(__name__).warning("No data to export to markdown")
            return
        baseline_df, target_df = analyzer.split_by_device(unique_df)
        new_failures_df = None
        new_passes_df = None
        if not baseline_df.empty and not target_df.empty:
            merged_df = analyzer.merge_results(baseline_df, target_df)
            new_failures_df, new_passes_df = analyzer.find_target_changes(merged_df)
        markdown_content = analyzer.generate_markdown_summary(unique_df, new_failures_df, new_passes_df)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logging.getLogger(__name__).info(f"Exported markdown report to {output_path}")

    def _generate_case_issue_df(self, analyzer: ResultAnalyzer) -> pd.DataFrame:
        if not analyzer.issue_tracker or not analyzer.issue_tracker.test_to_issues:
            return pd.DataFrame()
        tracker = analyzer.issue_tracker
        rows = []
        repo_name = tracker.repo_name
        for uniqname, issues in tracker.test_to_issues.items():
            issue_ids = [str(issue['id']) for issue in issues]
            issue_states = [issue['state'] for issue in issues]
            issue_labels_list = [', '.join(issue.get('labels', [])) for issue in issues]
            combined_ids = ', '.join(issue_ids)
            combined_states = ', '.join(issue_states)
            combined_labels = ' | '.join(issue_labels_list)
            if repo_name:
                urls = [f"https://github.com/{repo_name}/issues/{id}" for id in issue_ids]
                combined_urls = ', '.join(urls)
            else:
                combined_urls = ''
            rows.append({
                'Test Case (uniqname)': uniqname,
                'Issue IDs': combined_ids,
                'Issue States': combined_states,
                'Issue Labels': combined_labels,
                'Issue URLs': combined_urls,
            })
        return pd.DataFrame(rows)

# ---------- UT main ----------
def ut_main(args, issue_tracker: Optional[GitHubIssueTracker] = None):
    logger = logging.getLogger(__name__)
    start_time = time.time()
    if args.workers is None:
        args.workers = max(1, os.cpu_count() - 2)
        logger.info(f"Using {args.workers} workers (CPU count - 2)")
    extractor = TestDetailsExtractor()
    logger.info("Starting test extraction...")
    success = extractor.process(args.input, max_workers=args.workers)
    if not success:
        logger.error("No test cases found")
        return None
    logger.info(f"Found {len(extractor.test_cases)} test cases")
    analyzer = ResultAnalyzer(extractor.test_cases)
    if issue_tracker:
        analyzer.set_issue_tracker(issue_tracker)
    elif not args.no_github:
        github_repo = args.github_repo or os.environ.get('GITHUB_REPOSITORY')
        github_token = args.github_token or os.environ.get('GITHUB_TOKEN')
        if github_repo:
            logger.info(f"Initializing GitHub issue tracker for {github_repo}")
            new_tracker = GitHubIssueTracker(
                repo=github_repo,
                token=github_token,
                cache_path=args.github_issue_cache
            )
            if new_tracker.fetch_issues(
                state=args.github_issue_state,
                labels=args.github_labels,
                force_refresh=args.refresh_issues
            ):
                analyzer.set_issue_tracker(new_tracker)
                logger.info("GitHub issue tracking enabled")
            else:
                logger.warning("Failed to fetch GitHub issues, continuing without issue tracking")
        else:
            logger.info("GitHub repository not configured, skipping issue tracking")
    output_path = Path(args.output)
    exporter = ReportExporter()
    if output_path.suffix.lower() in [".xlsx", ".xls"]:
        exporter.export_excel(analyzer, output_path)
    else:
        exporter.export_csv(analyzer, output_path)
    if args.markdown:
        if args.markdown_output:
            markdown_path = Path(args.markdown_output)
        else:
            markdown_path = output_path.parent / f"{output_path.stem}_report.md"
        exporter.export_markdown(analyzer, markdown_path)
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"📊 Files processed: {extractor.stats['files_processed']}")
    print(f"🧪 Test cases found: {extractor.stats['test_cases_found']}")
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"📁 Output: {output_path}")
    if args.markdown:
        if args.markdown_output:
            print(f"📝 Markdown report: {markdown_path}")
        else:
            print(f"📝 Markdown report: {output_path.parent / f'{output_path.stem}_report.md'}")
    unique_df = analyzer.deduplicate_by_priority()
    if not unique_df.empty:
        baseline_count = len(unique_df[unique_df["device"] == "baseline"])
        target_count = len(unique_df[unique_df["device"] == "target"])
        print(f"📱 Baseline tests: {baseline_count}, Target tests: {target_count}")
        file_summary = analyzer.generate_file_summary(unique_df)
        if not file_summary.empty:
            print(f"📂 Test files: {len(file_summary)}")
        baseline_df, target_df = analyzer.split_by_device(unique_df)
        if not baseline_df.empty and not target_df.empty:
            merged_df = analyzer.merge_results(baseline_df, target_df)
            new_failures_df, new_passes_df = analyzer.find_target_changes(merged_df)
            if not new_failures_df.empty:
                print(f"🚨 New failures: {len(new_failures_df)}")
            if not new_passes_df.empty:
                print(f"✨ New passes: {len(new_passes_df)}")
    print("=" * 60)

# ----------------------------------------------------------------------
# ENV mode (environment summary)
# ----------------------------------------------------------------------

# ---------- ENV constants ----------
ENV_EMOJI = {
    "test_tube": "🧪",
    "fire": "🔥",
    "wrench": "🔧",
    "computer": "🖥️",
    "brain": "🧠",
    "clipboard": "📋",
    "snake": "🐍",
    "package": "📦",
}
ENV_PACKAGES_OF_INTEREST = {
    "oneapi": "intel-cmplr-lib-rt",
    "triton": "triton-xpu",
}
ENV_DEPENDENCY_PACKAGES = {
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "torchao": "torchao",
}
ENV_DEPENDENCY_ENV_VARS = {
    "Transformers": "TRANSFORMERS_VERSION",
    "Timm": "TIMM_COMMIT_ID",
    "TorchBench": "TORCHBENCH_COMMIT_ID",
}

# ---------- ENV parsing helpers ----------
def env_strip_command_lines(raw_log: str) -> str:
    lines = raw_log.splitlines()
    cleaned = []
    for line in lines:
        if line.lstrip().startswith('+'):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def env_parse_collect_env_output(cleaned_text: str) -> Dict[str, Any]:
    data = {"_raw": cleaned_text}
    lines = cleaned_text.splitlines()
    in_gpu_driver = False
    in_gpu_detected = False
    in_cpu = False
    in_versions = False
    gpu_driver_lines = []
    gpu_detected_lines = []
    cpu_lines = []
    pip_packages = []
    for idx, line in enumerate(lines):
        line = line.rstrip()
        if line.startswith("Intel GPU driver version:"):
            in_gpu_driver, in_gpu_detected, in_cpu, in_versions = True, False, False, False
            gpu_driver_lines = [line]
            continue
        if line.startswith("Intel GPU models detected:"):
            in_gpu_detected, in_gpu_driver, in_cpu, in_versions = True, False, False, False
            gpu_detected_lines = [line]
            continue
        if line.startswith("CPU:"):
            in_cpu, in_gpu_driver, in_gpu_detected, in_versions = True, False, False, False
            cpu_lines = [line]
            continue
        if line.startswith("Versions of relevant libraries:"):
            in_versions, in_gpu_driver, in_gpu_detected, in_cpu = True, False, False, False
            continue
        if in_gpu_driver:
            gpu_driver_lines.append(line)
            if line == "" or (idx < len(lines)-1 and lines[idx+1] and not lines[idx+1].startswith("*")):
                in_gpu_driver = False
                data["gpu_driver_version"] = "\n".join(gpu_driver_lines)
        elif in_gpu_detected:
            gpu_detected_lines.append(line)
            if line == "" or (idx < len(lines)-1 and lines[idx+1] and not lines[idx+1].startswith("*")):
                in_gpu_detected = False
                data["gpu_detected"] = "\n".join(gpu_detected_lines)
        elif in_cpu:
            cpu_lines.append(line)
            if line == "":
                in_cpu = False
                data["cpu_info"] = "\n".join(cpu_lines)
        elif in_versions:
            if line.startswith("[pip3]"):
                match = re.match(r"\[pip3\] (\S+)==(\S+)", line)
                if match:
                    pip_packages.append((match.group(1), match.group(2)))
            if line == "":
                in_versions = False
                data["pip_packages"] = pip_packages
        else:
            if ": " in line and not line.startswith(" "):
                key, val = line.split(": ", 1)
                data[key.strip()] = val.strip()
    if "pip_packages" not in data:
        data["pip_packages"] = pip_packages
    if "cpu_info" in data:
        cpu_summary = {}
        for line in data["cpu_info"].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                cpu_summary[k.strip()] = v.strip()
        data["cpu_summary"] = cpu_summary
    return data

def env_parse_printenv_output(cleaned_text: str) -> Dict[str, str]:
    env = {}
    for line in cleaned_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key] = val
    return env

def env_get_pip_version(pip_packages: List[tuple], package_name: str) -> str:
    for pkg, ver in pip_packages:
        if pkg == package_name:
            return ver
    return "N/A"

def env_clean_version_string(version_str: str) -> str:
    if not version_str or version_str == "N/A":
        return version_str
    cleaned = re.sub(r'^version\s+', '', version_str, flags=re.IGNORECASE)
    return cleaned.strip()

def env_extract_gpu_device_info(gpu_detected_text: str) -> Tuple[str, str]:
    name = "Unknown"
    memory = "?"
    for line in gpu_detected_text.splitlines():
        if line.startswith("* [0]"):
            name_match = re.search(r"name='([^']+)'", line)
            if name_match:
                name = name_match.group(1)
            mem_match = re.search(r"total_memory=(\d+)MB", line)
            if mem_match:
                memory = mem_match.group(1)
            break
    return name, memory

def env_extract_gpu_driver_version(gpu_driver_text: str) -> str:
    for line in gpu_driver_text.splitlines():
        if "intel-opencl-icd:" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "N/A"

def env_extract_onednn_version(full_raw_log: str) -> str:
    for line in full_raw_log.splitlines():
        if "onednn_verbose" in line and "oneDNN" in line:
            match = re.search(r"oneDNN\s+(.+)", line)
            if match:
                return match.group(1).strip()
    return "N/A"

def env_extract_memory_info(cleaned_text: str) -> str:
    for line in cleaned_text.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return "N/A"

def env_extract_disk_info(cleaned_text: str) -> str:
    lines = cleaned_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().endswith('/'):
            fields = line.split()
            if len(fields) >= 5 and fields[-1] == '/':
                size = fields[1]
                avail = fields[3]
                return f"{avail} / {size}"
    return "N/A"

# ---------- ENV single summary ----------
def env_generate_single_summary(collect_data: Dict[str, Any], env_vars: Dict[str, str], raw_log: str) -> str:
    lines = []
    lines.append(f"# {ENV_EMOJI['test_tube']} Test Environment Summary")
    lines.append("")
    lines.append(f"### {ENV_EMOJI['wrench']} Core")
    lines.append(f"- **PyTorch version**: `{collect_data.get('PyTorch version', 'N/A')}`")
    pkgs = collect_data.get("pip_packages", [])
    oneapi_ver = env_get_pip_version(pkgs, ENV_PACKAGES_OF_INTEREST["oneapi"])
    lines.append(f"- **oneAPI**: `{oneapi_ver}`")
    onednn_ver = env_extract_onednn_version(raw_log)
    lines.append(f"- **oneDNN**: `{onednn_ver}`")
    triton_ver = env_get_pip_version(pkgs, ENV_PACKAGES_OF_INTEREST["triton"])
    lines.append(f"- **Triton**: `{triton_ver}`")
    if "gpu_driver_version" in collect_data:
        driver_ver = env_extract_gpu_driver_version(collect_data["gpu_driver_version"])
    else:
        driver_ver = "N/A"
    lines.append(f"- **Driver**: `{driver_ver}`")
    lines.append("")
    lines.append(f"### {ENV_EMOJI['package']} Dependencies")
    for name, pkg in ENV_DEPENDENCY_PACKAGES.items():
        ver = env_get_pip_version(pkgs, pkg)
        lines.append(f"- **{name}**: `{ver}`")
    for name, env_var in ENV_DEPENDENCY_ENV_VARS.items():
        val = env_vars.get(env_var, "N/A")
        lines.append(f"- **{name}**: `{val}`")
    lines.append("")
    lines.append(f"### {ENV_EMOJI['computer']} System")
    lines.append(f"- **OS**: `{collect_data.get('OS', 'N/A')}`")
    lines.append(f"- **Kernel**: `{collect_data.get('Kernel version', 'N/A')}`")
    lines.append(f"- **{ENV_EMOJI['snake']} Python version**: `{collect_data.get('Python version', 'N/A')}`")
    lines.append(f"- **GCC version**: `{collect_data.get('GCC version', 'N/A')}`")
    cmake_ver = env_clean_version_string(collect_data.get('CMake version', 'N/A'))
    lines.append(f"- **CMake version**: `{cmake_ver}`")
    memory = env_extract_memory_info(raw_log)
    lines.append(f"- **Memory**: `{memory}`")
    disk = env_extract_disk_info(raw_log)
    lines.append(f"- **Disk**: `{disk}`")
    lines.append("")
    lines.append(f"### {ENV_EMOJI['brain']} CPU & GPU")
    if "cpu_summary" in collect_data:
        cpu = collect_data["cpu_summary"]
        lines.append(f"- **CPU Model**: `{cpu.get('Model name', 'N/A')}`")
        lines.append(f"- **CPU(s)**: `{cpu.get('CPU(s)', 'N/A')}`")
        lines.append(f"- **Architecture**: `{cpu.get('Architecture', 'N/A')}`")
    else:
        lines.append("- **CPU Model**: N/A")
    if "gpu_detected" in collect_data:
        gpu_name, gpu_mem = env_extract_gpu_device_info(collect_data["gpu_detected"])
        lines.append(f"- **GPU Model**: `{gpu_name} – {gpu_mem} MB`")
    else:
        lines.append("- **GPU Model**: N/A")
    ze_mask = env_vars.get("ZE_AFFINITY_MASK", "N/A")
    if ze_mask != "N/A" and ze_mask.strip():
        gpu_indices = [idx.strip() for idx in ze_mask.split(',') if idx.strip()]
        count = len(gpu_indices)
        lines.append(f"- **GPU(s)**: `{count} ({', '.join(gpu_indices)})`")
    else:
        lines.append("- **GPU(s)**: N/A")
    lines.append("")
    lines.append("<details>")
    lines.append(f"<summary><b>{ENV_EMOJI['clipboard']} Full combined log</b></summary>")
    lines.append("")
    lines.append("```")
    lines.append(raw_log.rstrip())
    lines.append("```")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)

def _env_format_value(val: Any) -> str:
    if val is None or val == "N/A":
        return "N/A"
    return str(val)

def _env_add_comparison_row(lines: List[str], description: str, target_val: Any, baseline_val: Any, fmt: str = "`{}`") -> None:
    t_str = fmt.format(_env_format_value(target_val))
    b_str = fmt.format(_env_format_value(baseline_val))
    marker = "" if t_str == b_str else " 🔄"
    lines.append(f"| **{description}** | {t_str} | {b_str} |{marker}")

def env_generate_comparison_summary(
    baseline_collect: Dict[str, Any],
    target_collect: Dict[str, Any],
    baseline_env: Dict[str, str],
    target_env: Dict[str, str],
    raw_baseline: str,
    raw_target: str
) -> str:
    lines = []
    lines.append(f"# {ENV_EMOJI['test_tube']} Test Environment Comparison")
    lines.append("")
    lines.append(f"| | {ENV_EMOJI['fire']} Target | {ENV_EMOJI['fire']} Baseline |")
    lines.append("| --- | --- | --- |")
    def pkg_ver(data, pkg):
        return env_get_pip_version(data.get("pip_packages", []), pkg)
    lines.append(f"| **{ENV_EMOJI['wrench']} Core** | | |")
    _env_add_comparison_row(lines, "PyTorch version",
                            target_collect.get('PyTorch version'),
                            baseline_collect.get('PyTorch version'))
    _env_add_comparison_row(lines, "oneAPI",
                            pkg_ver(target_collect, ENV_PACKAGES_OF_INTEREST["oneapi"]),
                            pkg_ver(baseline_collect, ENV_PACKAGES_OF_INTEREST["oneapi"]))
    _env_add_comparison_row(lines, "oneDNN",
                            env_extract_onednn_version(raw_target),
                            env_extract_onednn_version(raw_baseline))
    _env_add_comparison_row(lines, "Triton",
                            pkg_ver(target_collect, ENV_PACKAGES_OF_INTEREST["triton"]),
                            pkg_ver(baseline_collect, ENV_PACKAGES_OF_INTEREST["triton"]))
    t_driver = env_extract_gpu_driver_version(target_collect.get("gpu_driver_version", "")) if "gpu_driver_version" in target_collect else "N/A"
    b_driver = env_extract_gpu_driver_version(baseline_collect.get("gpu_driver_version", "")) if "gpu_driver_version" in baseline_collect else "N/A"
    _env_add_comparison_row(lines, "Driver", t_driver, b_driver)
    lines.append(f"| **{ENV_EMOJI['package']} Dependencies** | | |")
    for name, pkg in ENV_DEPENDENCY_PACKAGES.items():
        _env_add_comparison_row(lines, name,
                                pkg_ver(target_collect, pkg),
                                pkg_ver(baseline_collect, pkg))
    for name, env_var in ENV_DEPENDENCY_ENV_VARS.items():
        _env_add_comparison_row(lines, name,
                                target_env.get(env_var, "N/A"),
                                baseline_env.get(env_var, "N/A"))
    lines.append(f"| **{ENV_EMOJI['computer']} System** | | |")
    _env_add_comparison_row(lines, "OS",
                            target_collect.get('OS'),
                            baseline_collect.get('OS'))
    _env_add_comparison_row(lines, "Kernel",
                            target_collect.get('Kernel version'),
                            baseline_collect.get('Kernel version'))
    _env_add_comparison_row(lines, "Python version",
                            target_collect.get('Python version'),
                            baseline_collect.get('Python version'))
    _env_add_comparison_row(lines, "GCC version",
                            target_collect.get('GCC version'),
                            baseline_collect.get('GCC version'))
    cmake_t = env_clean_version_string(target_collect.get('CMake version', 'N/A'))
    cmake_b = env_clean_version_string(baseline_collect.get('CMake version', 'N/A'))
    _env_add_comparison_row(lines, "CMake version", cmake_t, cmake_b)
    mem_t = env_extract_memory_info(raw_target)
    mem_b = env_extract_memory_info(raw_baseline)
    _env_add_comparison_row(lines, "Memory", mem_t, mem_b)
    disk_t = env_extract_disk_info(raw_target)
    disk_b = env_extract_disk_info(raw_baseline)
    _env_add_comparison_row(lines, "Disk", disk_t, disk_b)
    lines.append(f"| **{ENV_EMOJI['brain']} CPU & GPU** | | |")
    t_cpu = target_collect.get("cpu_summary", {})
    b_cpu = baseline_collect.get("cpu_summary", {})
    _env_add_comparison_row(lines, "CPU Model",
                            t_cpu.get('Model name'),
                            b_cpu.get('Model name'))
    _env_add_comparison_row(lines, "CPU(s)",
                            t_cpu.get('CPU(s)'),
                            b_cpu.get('CPU(s)'))
    _env_add_comparison_row(lines, "Architecture",
                            t_cpu.get('Architecture'),
                            b_cpu.get('Architecture'))
    if "gpu_detected" in target_collect and "gpu_detected" in baseline_collect:
        t_name, t_mem = env_extract_gpu_device_info(target_collect["gpu_detected"])
        b_name, b_mem = env_extract_gpu_device_info(baseline_collect["gpu_detected"])
        _env_add_comparison_row(lines, "GPU Model",
                                f"{t_name} – {t_mem} MB",
                                f"{b_name} – {b_mem} MB")
    else:
        _env_add_comparison_row(lines, "GPU Model", "N/A", "N/A")
    def format_gpus(mask):
        if mask == "N/A" or not mask or not mask.strip():
            return "N/A"
        indices = [idx.strip() for idx in mask.split(',') if idx.strip()]
        return f"{len(indices)} ({', '.join(indices)})"
    _env_add_comparison_row(lines, "GPU(s)",
                            format_gpus(target_env.get("ZE_AFFINITY_MASK", "N/A")),
                            format_gpus(baseline_env.get("ZE_AFFINITY_MASK", "N/A")))
    lines.append("")
    lines.append("<details>")
    lines.append(f"<summary><b>{ENV_EMOJI['clipboard']} Target full log</b></summary>")
    lines.append("")
    lines.append("```")
    lines.append(raw_target.rstrip())
    lines.append("```")
    lines.append("</details>")
    lines.append("")
    lines.append("<details>")
    lines.append(f"<summary><b>{ENV_EMOJI['clipboard']} Baseline full log</b></summary>")
    lines.append("")
    lines.append("```")
    lines.append(raw_baseline.rstrip())
    lines.append("```")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)

# ---------- ENV main ----------
def env_main(args):
    if args.input:
        raw = Path(args.input).read_text()
        cleaned = env_strip_command_lines(raw)
        collect = env_parse_collect_env_output(cleaned)
        env = env_parse_printenv_output(cleaned)
        markdown = env_generate_single_summary(collect, env, raw)
    else:
        raw_base = Path(args.baseline).read_text()
        raw_tgt = Path(args.target).read_text()
        cleaned_base = env_strip_command_lines(raw_base)
        cleaned_tgt = env_strip_command_lines(raw_tgt)
        collect_base = env_parse_collect_env_output(cleaned_base)
        collect_tgt = env_parse_collect_env_output(cleaned_tgt)
        env_base = env_parse_printenv_output(cleaned_base)
        env_tgt = env_parse_printenv_output(cleaned_tgt)
        markdown = env_generate_comparison_summary(
            collect_base, collect_tgt,
            env_base, env_tgt,
            raw_base, raw_tgt
        )
    Path(args.output).write_text(markdown)
    print(f"Summary written to {args.output}")

# ----------------------------------------------------------------------
# FULL mode: run all three and merge markdown
# ----------------------------------------------------------------------

def full_main(args):
    # Setup logging
    setup_logging(debug=args.debug)

    # GitHub tracker (shared across components)
    issue_tracker = None
    if not args.no_github:
        github_repo = args.github_repo or os.environ.get('GITHUB_REPOSITORY')
        github_token = args.github_token or os.environ.get('GITHUB_TOKEN')
        if github_repo:
            logging.getLogger(__name__).info(f"Initializing GitHub issue tracker for {github_repo}")
            issue_tracker = GitHubIssueTracker(
                repo=github_repo,
                token=github_token,
                cache_path=args.github_issue_cache
            )
            if not issue_tracker.fetch_issues(
                state=args.github_issue_state,
                labels=args.github_labels,
                force_refresh=args.refresh_issues
            ):
                logging.getLogger(__name__).warning("Failed to fetch GitHub issues, continuing without issue tracking")
                issue_tracker = None
        else:
            logging.getLogger(__name__).info("GitHub repository not configured, skipping issue tracking")

    # Collect markdown parts
    parts = []

    # E2E
    if args.e2e_target_dir and args.e2e_baseline_dir and args.e2e_output:
        # We need to run e2e analysis and capture its markdown.
        # We'll create a dummy args object and call e2e_main in a way that returns markdown.
        # Since e2e_main writes markdown to file, we'll redirect to a temp file or modify e2e_main to return string.
        # Simpler: we can reuse e2e functions directly.
        print("\n--- Running E2E analysis ---")
        target_files = e2e_find_result_files(args.e2e_target_dir)
        baseline_files = e2e_find_result_files(args.e2e_baseline_dir)
        if target_files and baseline_files:
            target_acc = e2e_load_results(target_files, "accuracy")
            target_perf = e2e_load_results(target_files, "performance")
            baseline_acc = e2e_load_results(baseline_files, "accuracy")
            baseline_perf = e2e_load_results(baseline_files, "performance")
            acc_merged = e2e_merge_accuracy(target_acc, baseline_acc)
            perf_merged = e2e_merge_performance(target_perf, baseline_perf)
            combined_summary = e2e_generate_all_summaries(acc_merged, perf_merged)
            details = e2e_combine_results(acc_merged, perf_merged)
            combined_summary.rename(columns={k: v for k, v in E2E_COLUMN_RENAME_MAP.items() if k in combined_summary.columns}, inplace=True)
            details.rename(columns={k: v for k, v in E2E_COLUMN_RENAME_MAP.items() if k in details.columns}, inplace=True)
            e2e_md = e2e_generate_markdown_string(combined_summary, details)
            if e2e_md.strip():
                parts.append("# E2E Results\n" + e2e_md)
        else:
            print("E2E: no input files found, skipping.")

    # UT
    if args.ut_input:
        print("\n--- Running UT analysis ---")
        # We'll create a Namespace with required attributes
        ut_args = argparse.Namespace(
            input=args.ut_input,
            output=args.ut_output or "ut_comparison.xlsx",
            markdown=False,
            markdown_output=None,
            workers=args.workers,
            debug=args.debug,
            github_repo=args.github_repo,
            github_token=args.github_token,
            github_issue_state=args.github_issue_state,
            github_labels=args.github_labels,
            no_github=args.no_github,
            github_issue_cache=args.github_issue_cache,
            refresh_issues=args.refresh_issues,
        )
        # We need to capture the markdown without writing to file. We'll modify ut_main to optionally return markdown string.
        # For simplicity, we'll call ut_main and let it write to a temp file, then read back.
        # But since we have the analyzer, we can compute the markdown directly.
        extractor = TestDetailsExtractor()
        success = extractor.process(ut_args.input, max_workers=ut_args.workers)
        if success:
            analyzer = ResultAnalyzer(extractor.test_cases)
            if issue_tracker:
                analyzer.set_issue_tracker(issue_tracker)
            unique_df = analyzer.deduplicate_by_priority()
            baseline_df, target_df = analyzer.split_by_device(unique_df)
            new_failures_df, new_passes_df = None, None
            if not baseline_df.empty and not target_df.empty:
                merged_df = analyzer.merge_results(baseline_df, target_df)
                new_failures_df, new_passes_df = analyzer.find_target_changes(merged_df)
            ut_md = analyzer.generate_markdown_summary(unique_df, new_failures_df, new_passes_df)
            if ut_md.strip():
                parts.append("# UT Results\n" + ut_md)
        else:
            print("UT: no test cases found, skipping.")

    # ENV
    env_md = ""
    if args.env_input:
        print("\n--- Running ENV analysis (single) ---")
        raw = Path(args.env_input).read_text()
        cleaned = env_strip_command_lines(raw)
        collect = env_parse_collect_env_output(cleaned)
        env = env_parse_printenv_output(cleaned)
        env_md = env_generate_single_summary(collect, env, raw)
    elif args.env_baseline and args.env_target:
        print("\n--- Running ENV analysis (comparison) ---")
        raw_base = Path(args.env_baseline).read_text()
        raw_tgt = Path(args.env_target).read_text()
        cleaned_base = env_strip_command_lines(raw_base)
        cleaned_tgt = env_strip_command_lines(raw_tgt)
        collect_base = env_parse_collect_env_output(cleaned_base)
        collect_tgt = env_parse_collect_env_output(cleaned_tgt)
        env_base = env_parse_printenv_output(cleaned_base)
        env_tgt = env_parse_printenv_output(cleaned_tgt)
        env_md = env_generate_comparison_summary(
            collect_base, collect_tgt,
            env_base, env_tgt,
            raw_base, raw_tgt
        )
    if env_md.strip():
        parts.append("# Environment Summary\n" + env_md)

    # Combine and write final markdown
    if parts:
        final_md = "\n\n".join(parts)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(final_md)
        print(f"\nCombined markdown written to {args.output}")
    else:
        print("\nNo data from any component, no output written.")

# ----------------------------------------------------------------------
# Main dispatcher
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified test analysis tool")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommand')

    # e2e subcommand
    parser_e2e = subparsers.add_parser('e2e', help='Compare inductor end‑to‑end performance/accuracy results')
    parser_e2e.add_argument("-t", "--target_dir", required=True, help="Directory containing target result CSV files")
    parser_e2e.add_argument("-b", "--baseline_dir", required=True, help="Directory containing baseline result CSV files")
    parser_e2e.add_argument("-o", "--output", required=True, help="Output file name (.xlsx or .csv)")
    parser_e2e.add_argument("-m", "--markdown", help="Generate a Markdown summary report and save to this file")

    # ut subcommand
    parser_ut = subparsers.add_parser('ut', help='Compare unit test JUnit XML results')
    parser_ut.add_argument("-i", "--input", nargs="+", required=True, help="XML file paths, directories, or glob patterns")
    parser_ut.add_argument("-o", "--output", default="test_comparison.xlsx", help="Output file path (.xlsx or .csv)")
    parser_ut.add_argument("-m", "--markdown", action="store_true", help="Generate a markdown summary file")
    parser_ut.add_argument("--markdown-output", help="Output path for markdown file (default: {output_stem}_report.md)")
    parser_ut.add_argument("-w", "--workers", type=int, help="Number of parallel workers (default: CPU count - 2)")
    parser_ut.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser_ut.add_argument("--github-repo", default="intel/torch-xpu-ops", help="GitHub repository in format 'owner/repo'")
    parser_ut.add_argument("--github-token", help="GitHub personal access token")
    parser_ut.add_argument("--github-issue-state", default="open", choices=["open", "closed", "all"], help="State of issues to fetch")
    parser_ut.add_argument("--github-labels", nargs="+", default="skipped", help="Filter issues by labels")
    parser_ut.add_argument("--no-github", action="store_true", help="Disable GitHub integration")
    parser_ut.add_argument("--github-issue-cache", default="selected_issues.json", help="Path to cache file for GitHub issues")
    parser_ut.add_argument("--refresh-issues", action="store_true", help="Force refresh GitHub issues even if cache exists")

    # env subcommand
    parser_env = subparsers.add_parser('env', help='Generate environment summary from combined log')
    group_env = parser_env.add_mutually_exclusive_group(required=True)
    group_env.add_argument("--input", "-i", help="Single combined log file (normal summary)")
    group_env.add_argument("--baseline", help="Baseline log file (for comparison)")
    parser_env.add_argument("--target", help="Target log file (required with --baseline)")
    parser_env.add_argument("--output", "-o", required=True, help="Output Markdown file")

    # full subcommand
    parser_full = subparsers.add_parser('full', help='Run e2e, ut, and env analyses and merge markdown reports')
    parser_full.add_argument("--e2e-target-dir", help="E2E target directory")
    parser_full.add_argument("--e2e-baseline-dir", help="E2E baseline directory")
    parser_full.add_argument("--e2e-output", help="E2E output file (optional, only for data files)")
    parser_full.add_argument("--ut-input", nargs="+", help="UT input paths")
    parser_full.add_argument("--ut-output", help="UT output file (optional)")
    parser_full.add_argument("--env-input", help="ENV single log file")
    parser_full.add_argument("--env-baseline", help="ENV baseline log file")
    parser_full.add_argument("--env-target", help="ENV target log file")
    parser_full.add_argument("-o", "--output", required=True, help="Output merged markdown file")
    parser_full.add_argument("-w", "--workers", type=int, help="Number of parallel workers (default: CPU count - 2)")
    parser_full.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser_full.add_argument("--github-repo", default="intel/torch-xpu-ops", help="GitHub repository")
    parser_full.add_argument("--github-token", help="GitHub personal access token")
    parser_full.add_argument("--github-issue-state", default="open", choices=["open", "closed", "all"], help="State of issues to fetch")
    parser_full.add_argument("--github-labels", nargs="+", default="skipped", help="Filter issues by labels")
    parser_full.add_argument("--no-github", action="store_true", help="Disable GitHub integration")
    parser_full.add_argument("--github-issue-cache", default="selected_issues.json", help="Path to cache file for GitHub issues")
    parser_full.add_argument("--refresh-issues", action="store_true", help="Force refresh GitHub issues even if cache exists")

    args = parser.parse_args()

    # Setup logging
    debug = getattr(args, 'debug', False)
    setup_logging(debug)

    if args.command == 'e2e':
        e2e_main(args)
    elif args.command == 'ut':
        ut_main(args)
    elif args.command == 'env':
        env_main(args)
    elif args.command == 'full':
        full_main(args)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.getLogger(__name__).error(f"Unexpected error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)
