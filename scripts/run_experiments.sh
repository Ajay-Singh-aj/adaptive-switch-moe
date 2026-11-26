#!/usr/bin/env bash
# Reproduce every run reported in the README.
# Usage: bash scripts/run_experiments.sh [python-executable] [parallel-jobs]
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${1:-python}"
JOBS="${2:-3}"
STEPS=5000
SEEDS="0 1 2"

jobs=()
for s in $SEEDS; do
  for cf in 1.0 1.25; do
    jobs+=("--moe baseline --capacity-factor $cf --seed $s --run-name baseline_cf${cf}_s$s")
    jobs+=("--moe adaptive --capacity-factor $cf --seed $s --run-name adaptive_cf${cf}_s$s")
  done
  # ablations at capacity factor 1.0: remove one component at a time
  jobs+=("--moe adaptive --seed $s --no-reassign          --run-name ablate_no_reassign_s$s")
  jobs+=("--moe adaptive --seed $s --no-importance-priority --run-name ablate_no_importance_s$s")
  jobs+=("--moe adaptive --seed $s --simbal-coef 0          --run-name ablate_no_simbal_s$s")
done

printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} sh -c "$PY train.py --steps $STEPS {}"
"$PY" scripts/summarize.py
