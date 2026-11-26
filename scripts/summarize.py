"""Aggregate runs in results/runs into results/summary.md and results/figures/*.png.

Runs are grouped by run name with the trailing `_s<seed>` removed; each table
row reports mean +/- std across seeds of the final evaluation.
"""

from __future__ import annotations

import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
FIGS = ROOT / "results" / "figures"

LABELS = {
    "baseline_cf1.0": "Switch baseline (CF 1.0)",
    "adaptive_cf1.0": "Adaptive (CF 1.0)",
    "baseline_cf1.25": "Switch baseline (CF 1.25)",
    "adaptive_cf1.25": "Adaptive (CF 1.25)",
    "ablate_no_reassign": "Adaptive w/o NTLB reassignment",
    "ablate_no_importance": "Adaptive w/o importance priority",
    "ablate_no_simbal": "Adaptive w/o SimBal",
}


def load_runs():
    groups = defaultdict(list)
    for summary in sorted(RUNS.glob("*/summary.json")):
        s = json.loads(summary.read_text())
        with open(summary.parent / "metrics.csv") as f:
            s["curve"] = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        groups[re.sub(r"_s\d+$", "", s["run_name"])].append(s)
    return groups


def fmt(values, digits=4, pct=False):
    scale = 100 if pct else 1
    vals = [v * scale for v in values]
    mean = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    suffix = "%" if pct else ""
    return f"{mean:.{digits}f} ± {sd:.{digits}f}{suffix}"


def write_table(groups):
    lines = [
        "| Model | Seeds | Val loss | Val drop rate | Train drop rate | Train reassign rate |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in LABELS.items():
        runs = groups.get(key)
        if not runs:
            continue
        final = [r["final"] for r in runs]
        lines.append(
            f"| {label} | {len(runs)} | {fmt([f['val_loss'] for f in final])} "
            f"| {fmt([f['val_drop_rate'] for f in final], 2, True)} "
            f"| {fmt([f['train_drop_rate'] for f in final], 2, True)} "
            f"| {fmt([f['train_reassign_rate'] for f in final], 2, True)} |"
        )
    table = "\n".join(lines)
    (ROOT / "results" / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


def mean_curve(runs, key):
    steps = [row["step"] for row in runs[0]["curve"]]
    values = [st.mean(r["curve"][i][key] for r in runs) for i in range(len(steps))]
    return steps, values


def plot(groups, keys, metric, ylabel, filename, pct=False):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key in keys:
        if key not in groups:
            continue
        steps, vals = mean_curve(groups[key], metric)
        ax.plot(steps, [v * (100 if pct else 1) for v in vals], marker="o", ms=3, label=LABELS[key])
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / filename, dpi=150)
    plt.close(fig)


def main():
    groups = load_runs()
    if not groups:
        raise SystemExit(f"No runs found in {RUNS}")
    FIGS.mkdir(parents=True, exist_ok=True)
    write_table(groups)
    main_keys = ["baseline_cf1.0", "adaptive_cf1.0", "baseline_cf1.25", "adaptive_cf1.25"]
    plot(groups, main_keys, "val_loss", "Validation loss", "val_loss.png")
    plot(groups, main_keys, "train_drop_rate", "Dropped tokens (%)", "drop_rate.png", pct=True)
    plot(groups, ["adaptive_cf1.0", "ablate_no_reassign", "ablate_no_importance", "ablate_no_simbal"],
         "val_loss", "Validation loss", "ablations_val_loss.png")


if __name__ == "__main__":
    main()
