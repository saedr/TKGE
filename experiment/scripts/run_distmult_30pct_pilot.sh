#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}/experiment:${PYTHONPATH:-}"
RESULTS_ROOT="experiment/results/raw"
TAG="pilot_distmult_30pct"
CONFIG="experiment/configs/smoke.yaml"
SEEDS=(11 22 33)
CONDS=(Original Random Structured-low Structured-high)

for seed in "${SEEDS[@]}"; do
  for condition in "${CONDS[@]}"; do
    python -m experiment.src.run_condition \
      --config "$CONFIG" \
      --condition "$condition" \
      --results-root "$RESULTS_ROOT" \
      --seed "$seed" \
      --budget 0.30 \
      --model DistMult \
      --tag "$TAG/seed_${seed}"
  done
done
