#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024, 2025, 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Run every example script in this folder and record what happened.

Each ``NN_*.py`` file is executed, in numeric order, with the Python
interpreter of the current environment (``sys.executable``). As soon as one
example exits with a non-zero return code, crashes, or times out, the run
stops, the failing example's output is printed to the terminal, and the
process exits with a non-zero status.

Whether the run completes fully or stops early, a report is written to a new,
timestamped folder under ``run_results/`` containing:

  * ``report.md``    - human readable summary (status, duration, key metric
                        lines such as loss/accuracy/noise values found in the
                        output of each example).
  * ``metrics.json``  - the same information in a machine readable form, so
                        that it can be diffed against other runs (see
                        ``compare_runs.py``).
  * ``logs/*.log``    - the full stdout/stderr of every example that was run.

The top level ``run_results/history.jsonl`` file gets one line appended per
run so trends can be tracked over time without opening every report.

Usage:
    python examples/run_examples.py
    python examples/run_examples.py --list
    python examples/run_examples.py --exclude "24_bert_on_squad.py" "36_*"
    python examples/run_examples.py --only "0*.py" --timeout 900
    python examples/run_examples.py --continue-on-error
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXAMPLES_DIR / "run_results"
SELF_NAME = Path(__file__).name
COMPANION_SCRIPTS = {SELF_NAME, "compare_runs.py"}

# Keywords used to pick out the lines worth keeping for later comparison
# (loss curves, accuracy, injected noise levels, drift, timing, ...).
METRIC_KEYWORDS = (
    "loss",
    "accuracy",
    "acc",
    "error",
    "noise",
    "drift",
    "snr",
    "mse",
    "rmse",
    "mae",
    "perplexity",
    "score",
    "f1",
    "exact_match",
    "reward",
    "time",
)
NUMBER_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")


def discover_examples(only, exclude):
    """Return the sorted list of example scripts, honoring filters."""
    scripts = sorted(
        p for p in EXAMPLES_DIR.glob("*.py") if p.name not in COMPANION_SCRIPTS
    )
    if only:
        scripts = [p for p in scripts if any(fnmatch.fnmatch(p.name, pat) for pat in only)]
    if exclude:
        scripts = [p for p in scripts if not any(fnmatch.fnmatch(p.name, pat) for pat in exclude)]
    return scripts


def extract_metric_lines(text):
    """Pull out lines that look like they report a loss/accuracy/noise/etc value."""
    lines = []
    for line in text.splitlines():
        low = line.lower()
        if any(keyword in low for keyword in METRIC_KEYWORDS) and NUMBER_RE.search(line):
            lines.append(line.strip())
    return lines


def run_example(script, python_exe, timeout, env, log_path):
    """Execute a single example script and capture everything about the run."""
    start = time.time()
    output = ""
    try:
        proc = subprocess.run(
            [python_exe, str(script)],
            cwd=str(script.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
            check=False,
        )
        output = proc.stdout or ""
        status = "success" if proc.returncode == 0 else "failed"
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        status = "timeout"
        returncode = None
    duration = time.time() - start

    log_path.write_text(output, encoding="utf-8")
    lines = output.splitlines()
    return {
        "script": script.name,
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(duration, 2),
        "metric_lines": extract_metric_lines(output),
        "log_file": str(log_path.relative_to(RESULTS_DIR)),
        "tail": lines[-40:],
    }


def write_report(run_dir, run_id, python_exe, results, pending, stopped_early):
    """Write metrics.json + report.md for the (possibly partial) run."""
    total = len(results) + len(pending)
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = [r for r in results if r["status"] != "success"]

    summary = {
        "run_id": run_id,
        "python": python_exe,
        "examples_total": total,
        "examples_run": len(results),
        "examples_succeeded": succeeded,
        "examples_failed": len(failed),
        "stopped_early": stopped_early,
        "pending_not_run": [p.name for p in pending],
        "results": results,
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        f"# Example run report - {run_id}",
        "",
        f"* Python: `{python_exe}`",
        f"* Examples discovered: {total}",
        f"* Examples run: {len(results)}",
        f"* Succeeded: {succeeded}",
        f"* Failed: {len(failed)}",
        f"* Stopped early: {'yes' if stopped_early else 'no'}",
        "",
        "## Results",
        "",
        "| # | Example | Status | Duration (s) | Exit code | Log |",
        "|---|---------|--------|---------------|-----------|-----|",
    ]
    for i, r in enumerate(results, start=1):
        report_lines.append(
            f"| {i} | {r['script']} | {r['status']} | {r['duration_seconds']} "
            f"| {r['returncode']} | `{r['log_file']}` |"
        )
    for i, p in enumerate(pending, start=len(results) + 1):
        report_lines.append(f"| {i} | {p.name} | not_run | - | - | - |")

    report_lines.append("")
    report_lines.append("## Key metric values captured per example")
    report_lines.append("")
    for r in results:
        report_lines.append(f"### {r['script']} ({r['status']})")
        if r["metric_lines"]:
            report_lines.append("```")
            report_lines.extend(r["metric_lines"])
            report_lines.append("```")
        else:
            report_lines.append("_No loss/accuracy/noise/etc. values detected in output._")
        report_lines.append("")

    if failed:
        report_lines.append("## Failure details")
        report_lines.append("")
        for r in failed:
            report_lines.append(f"### {r['script']}")
            report_lines.append(f"Status: {r['status']}, exit code: {r['returncode']}")
            report_lines.append("```")
            report_lines.extend(r["tail"])
            report_lines.append("```")
            report_lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    history_path = RESULTS_DIR / "history.jsonl"
    history_entry = {
        "run_id": run_id,
        "timestamp": run_id,
        "examples_total": total,
        "examples_run": len(results),
        "examples_succeeded": succeeded,
        "examples_failed": len(failed),
        "stopped_early": stopped_early,
        "run_dir": str(run_dir.relative_to(RESULTS_DIR)),
    }
    with open(history_path, "a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(history_entry) + "\n")

    return report_path, metrics_path


def main():
    parser = argparse.ArgumentParser(
        description="Run all aihwkit examples and report the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Notes:\n"
            "  Some examples download datasets/models on first use (MNIST, CIFAR10,\n"
            "  SQuAD, WikiText, ...) and can take a long time or need internet\n"
            "  access; 20_mnist_ddp.py spawns multiple processes and expects\n"
            "  multiple GPUs; 17_resnet34_imagenet_conversion_to_analog.py expects a\n"
            "  local ImageNet copy. Use --exclude/--only to shape a run to your\n"
            "  environment, or --list to preview the execution order first."
        ),
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Python executable to run examples with."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="PATTERN",
        help="Only run examples whose filename matches one of these glob patterns.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="PATTERN",
        help="Skip examples whose filename matches one of these glob patterns.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-example timeout in seconds (default: no timeout).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining examples after a failure instead of stopping.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the examples that would be run, in order, and exit.",
    )
    args = parser.parse_args()

    scripts = discover_examples(args.only, args.exclude)
    if not scripts:
        print("No example scripts matched the given filters.")
        sys.exit(1)

    if args.list:
        for script in scripts:
            print(script.name)
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{run_id}"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")  # avoid blocking on plt.show()

    print(f"Running {len(scripts)} example(s) with {args.python}")
    print(f"Results will be written to {run_dir}")
    print()

    results = []
    stopped_early = False
    for index, script in enumerate(scripts):
        print(f"[{index + 1}/{len(scripts)}] {script.name} ... ", end="", flush=True)
        log_path = logs_dir / f"{script.stem}.log"
        result = run_example(script, args.python, args.timeout, env, log_path)
        results.append(result)

        if result["status"] == "success":
            print(f"OK ({result['duration_seconds']}s)")
        else:
            print(f"FAILED ({result['status']}, exit code={result['returncode']})")
            print()
            print(f"===== {script.name}: last output =====")
            print("\n".join(result["tail"]))
            print("=====")
            print(f"Full log: {log_path}")
            print()

            if not args.continue_on_error:
                stopped_early = True
                pending = scripts[index + 1 :]
                report_path, _ = write_report(
                    run_dir, run_id, args.python, results, pending, stopped_early
                )
                print(f"Stopped after failure. Partial report written to {report_path}")
                sys.exit(1)

    pending = []
    report_path, metrics_path = write_report(
        run_dir, run_id, args.python, results, pending, stopped_early
    )

    failed = sum(1 for r in results if r["status"] != "success")
    print()
    print(f"Finished: {len(results) - failed}/{len(results)} examples succeeded.")
    print(f"Report:  {report_path}")
    print(f"Metrics: {metrics_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
