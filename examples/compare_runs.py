#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024, 2025, 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Compare the metric values captured by two runs of ``run_examples.py``.

By default the two most recent runs under ``run_results/`` are compared. Pass
explicit run directories (or run ids) to compare any two runs instead.

Usage:
    python examples/compare_runs.py
    python examples/compare_runs.py run_20260101_120000 run_20260102_090000
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "run_results"


def resolve_run(identifier):
    """Accept a run id, a run folder name, or a full/relative path."""
    candidates = [
        Path(identifier),
        RESULTS_DIR / identifier,
        RESULTS_DIR / f"run_{identifier}",
    ]
    for candidate in candidates:
        metrics = candidate / "metrics.json"
        if metrics.is_file():
            return candidate, json.loads(metrics.read_text(encoding="utf-8"))
    raise SystemExit(f"Could not find a run (with metrics.json) matching '{identifier}'")


def latest_runs(count=2):
    runs = sorted((p for p in RESULTS_DIR.glob("run_*") if (p / "metrics.json").is_file()))
    if len(runs) < count:
        raise SystemExit(
            f"Need at least {count} completed runs under {RESULTS_DIR} to compare, found {len(runs)}."
        )
    return runs[-count:]


def index_by_script(summary):
    return {r["script"]: r for r in summary["results"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", nargs="?", help="Older run id/folder (default: second-to-last run).")
    parser.add_argument("run_b", nargs="?", help="Newer run id/folder (default: last run).")
    args = parser.parse_args()

    if args.run_a and args.run_b:
        dir_a, summary_a = resolve_run(args.run_a)
        dir_b, summary_b = resolve_run(args.run_b)
    elif not args.run_a and not args.run_b:
        dir_a, dir_b = latest_runs(2)
        summary_a = json.loads((dir_a / "metrics.json").read_text(encoding="utf-8"))
        summary_b = json.loads((dir_b / "metrics.json").read_text(encoding="utf-8"))
    else:
        raise SystemExit("Provide either zero or two run identifiers.")

    print(f"Comparing {dir_a.name} -> {dir_b.name}\n")

    results_a = index_by_script(summary_a)
    results_b = index_by_script(summary_b)
    all_scripts = sorted(set(results_a) | set(results_b))

    for script in all_scripts:
        ra = results_a.get(script)
        rb = results_b.get(script)
        if ra is None:
            print(f"[{script}] only present in {dir_b.name} (status={rb['status']})")
            continue
        if rb is None:
            print(f"[{script}] only present in {dir_a.name} (status={ra['status']})")
            continue

        status_changed = ra["status"] != rb["status"]
        header = f"[{script}] {ra['status']} -> {rb['status']}" if status_changed else f"[{script}] {rb['status']}"
        print(header)

        lines_a = ra.get("metric_lines", [])
        lines_b = rb.get("metric_lines", [])
        if lines_a != lines_b:
            last_a = lines_a[-1] if lines_a else "(none)"
            last_b = lines_b[-1] if lines_b else "(none)"
            print(f"    last metric line before: {last_a}")
            print(f"    last metric line after:  {last_b}")
        print()


if __name__ == "__main__":
    main()
