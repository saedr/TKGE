# Confirmatory structured-missingness replication

This branch implements the frozen WN18RR/CoDEx-M gate. It retains the branch's
existing training setup (64 dimensions, 8 epochs, batch size 512, five tail
negatives, Adam at 0.01) and changes no scientific factor outside the frozen
grid.

## Frozen grid

- Datasets: WN18RR and CoDEx-M.
- Models: TransE, DistMult, ComplEx, RotatE.
- Conditions: Original, Random, entity-coverage `Structured-low`, and
  relation-frequency `Relation-low`.
- Deletion budget: 30%.
- Entity Structured-low sampling weights:
  `1/(min(original_degree(head), original_degree(tail)) + 5)`.
- Relation-low sampling weights:
  `1/(original_training_relation_frequency(relation) + 5)`.
- Every perturbed condition has deletion seeds 101, 202, 303 crossed with
  training seeds 11, 22, 33. Original has the three training seeds.
- A deletion is skipped if it would leave either incident training entity with
  no retained training edge. Validation and test data are never changed.
- All original test triples are evaluated in the tail direction. Filtering uses
  the complete original train+validation+test union, including deleted training
  triples. Rank ties are optimistic: rank is one plus the number of unfiltered
  candidates with strictly greater score.
- Low/mid/high coverage and relation-frequency slices use original-training-
  graph quartiles and are shared by every model, condition, and realization.

## Execution and resumption

Run one matrix chunk from the repository root:

```bash
PYTHONPATH=. python -u -m experiment.scripts.run_confirmatory_chunk \
  --dataset WN18RR --model TransE \
  --out confirmatory_raw/WN18RR_TransE.json
```

The JSON result is replaced atomically after every completed training/evaluation
run, and a sibling `.progress.log` is appended and flushed. Repeating the same
command validates the frozen metadata and skips every completed
`(condition, deletion_seed, training_seed)` key.

After all eight dataset/model chunks are complete:

```bash
PYTHONPATH=. python -m experiment.scripts.aggregate_confirmatory \
  --indir confirmatory_raw \
  --out confirmatory_analysis/confirmatory_summary.json \
  --report confirmatory_analysis/confirmatory_report.md \
  --tables-dir confirmatory_analysis/tables
```

The analysis uses 4,000 paired crossed hierarchical bootstrap replications.
Deletion-realization, training-seed, and within-slice query indices are
resampled and shared across models and mechanisms. This yields intervals for
mechanism means, degradation, differential degradation, model × mechanism
interactions, and model gaps. A reversal is reported only if the two
mechanism-specific model-gap intervals exclude zero in opposite directions.

## Pre-run audit fixes

The initial branch implementation had four execution/specification defects:

1. Direct script execution could not import `experiment` in GitHub Actions.
2. Evaluation silently used only the first 400 test triples.
3. A restarted chunk reran all already completed combinations.
4. Aggregation did not resample queries, omitted mid/high slices, and accepted
   point-crossing reversals without two direction-supporting intervals.

The runner, workflow, and aggregation on this branch correct those defects
without altering the frozen models, training setup, deletion mechanisms,
budget, seed grid, or evaluation direction.
