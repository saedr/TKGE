#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/experiment:${PYTHONPATH:-}"
for seed in 11 22 33; do
  for cond in Original Random Structured-low Structured-high; do
    python -m experiment.src.run_condition \
      --config experiment/configs/smoke.yaml \
      --condition "$cond" \
      --results-root experiment/results/raw \
      --seed "$seed" \
      --budget 0.10 \
      --model DistMult \
      --tag full_grid/DistMult/budget_10/seed_${seed}
  done
done
