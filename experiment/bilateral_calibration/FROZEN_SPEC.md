# Frozen falsification specification: bilateral construction-noise calibration

Frozen: 2026-09-08, before any pilot outcome is inspected.

This specification may be changed only for a clearly documented implementation bug that prevents the frozen experiment from being executed as written. Scientific choices, datasets, corruption rate, models, metrics, thresholds, and gates may not be changed after results are observed.

## Question

Does structured, model-produced KG construction noise create a **topology-specific calibration failure** beyond the effect of the same noisy calibration labels alone?

The target contrast is:

`BILATERAL - LABEL_ONLY`

where both conditions use the same corrupted calibration labels, but only BILATERAL trains the KGE on a graph containing false observed edges.

The clean benchmark graph is called the **reference graph**, not world truth.

## Datasets

Exactly two datasets:

1. **FB15k-237** — primary.
2. **WN18RR** — prespecified replication.

Source: the public AAAI-2024 CCA repository `nju-websoft/CCA`, pinned to commit:

`9bd43e9f4277d533727adad34ca480fe0779c70b`

Use its standard `train.txt`, `dev.txt`, and `test.txt` files. No alternative dataset may be introduced if a gate fails.

Reference-positive set `G*` is the union of all three official positive splits. A generated triple is treated as false only if it is absent from `G*`. This is a benchmark closed-world reference, not a claim that every unobserved real-world fact is false.

## Construction-error mechanism

No uniform random graph corruption is allowed as the construction-noise mechanism.

Use a separate **TransE hard-error generator** trained only on the clean official training graph.

Generator:

- embedding dimension: 64;
- epochs: 20;
- Adam learning rate: 1e-3;
- batch size: 4096;
- one negative per positive;
- margin-ranking objective, margin 1.0;
- generator RNG seed: `20260908`;
- negatives rejected if present in `G*`.

Generator adequacy check: on 5,000 deterministic test positive/negative pairs (or all test positives if fewer), the trained TransE must rank the positive above its relation-compatible negative at least 65% of the time. Otherwise `KILL-BENCHMARK`.

### Hard false additions

Noise rate is exactly **5% of the observed graph**. For a clean split with `n` positive triples, target false additions are:

`m = ceil(0.05 / 0.95 * n)`.

Generate false additions independently for:

- the official training split;
- calibration-fit half of official dev;
- calibration-evaluation half of official dev.

For each candidate source triple, choose head-versus-tail replacement by the frozen RNG. Candidate replacement entities are restricted to entities observed in that same head/tail position for the same relation in the **clean training graph**. Score valid candidates with the frozen TransE generator and take the highest-scoring candidate that:

- differs from the source triple;
- is absent from `G*`;
- has not already been generated for that split.

If the chosen side has no valid candidate, try the other side. Iterate through a frozen random permutation of source triples until `m` distinct false additions are obtained.

If fewer than 98% of the target additions can be generated on any required split, `KILL-BENCHMARK`. Do not switch corruption mechanisms.

False additions are **added**; clean positives are not deleted. This mirrors a KG construction pipeline that accepts spurious extracted/completed facts.

## Dev split

Deterministically shuffle official dev positives with RNG seed `20260908` and split 50/50:

- first half: calibration fit;
- second half: calibration evaluation.

Construction false additions are generated separately for each half at the same 5%-of-observed rate.

## Fixed binary candidate pools

Calibration is evaluated as triple correctness on fixed binary candidate pools.

For every clean positive in each dev half and test split, create one deterministic relation-compatible negative by replacing head or tail with an entity observed in that position for the same relation in clean training, rejecting anything in `G*`. If no relation-compatible negative exists, use the first valid entity found from a deterministic seed-shuffled global entity list. These evaluation negatives are **not construction errors**; they exist only to define a fixed binary calibration task.

Candidate pools:

### CLEAN dev pool
- clean dev positives: observed label 1, reference label 1;
- fixed negatives: observed label 0, reference label 0.

### Noisy dev pool used by LABEL_ONLY and BILATERAL
- all clean dev positives: observed label 1, reference label 1;
- generated false additions: observed label 1, reference label 0;
- fixed negatives: observed label 0, reference label 0.

LABEL_ONLY and BILATERAL must use byte-identical noisy calibration-fit and calibration-evaluation pools.

### Test pool
- clean official test positives: reference label 1;
- one fixed negative per positive: reference label 0.

The test pool is never corrupted and is byte-identical across all conditions.

## Downstream KGE

Exactly one downstream KGE architecture: **ComplEx**.

Three seeds: `11`, `23`, `37`.

For each dataset and seed:

- embedding dimension: 64;
- epochs: 30;
- batch size: 4096;
- Adam learning rate: 1e-3;
- four sampled negatives per training positive;
- binary logistic/BCE training objective;
- sampled negatives rejected if present in `G*`;
- CPU is sufficient; no hyperparameter search.

No DistMult, TransE, RotatE, or other downstream model may be added after seeing outcomes.

## Experimental conditions

For each dataset and downstream seed:

### CLEAN
- train ComplEx on clean training graph;
- fit calibrator on CLEAN calibration-fit pool using clean observed labels.

### LABEL_ONLY
- reuse the **same clean-trained ComplEx checkpoint** as CLEAN;
- fit a separate calibrator on the noisy calibration-fit pool using noisy observed labels.

### BILATERAL
- train ComplEx from the same seed on clean training positives **plus frozen generated false training additions**;
- fit calibrator on the exact same noisy calibration-fit pool and noisy observed labels used by LABEL_ONLY.

Thus LABEL_ONLY versus BILATERAL holds calibration-label corruption fixed and changes only whether false edges contaminated representation learning.

## Calibrator

Exactly **Platt scaling**:

`p(y=1 | s) = sigmoid(a*s + b)`

Fit `a,b` by logistic loss on the designated calibration-fit candidate pool. No isotonic regression, vector scaling, KGE Calibrator, conformal method, or alternative calibration method may be substituted after seeing results.

## Metrics

Primary calibration metric: **ECE-10**, ten equal-width probability bins on [0,1].

Required supporting metrics:

- Brier score;
- negative log likelihood, probabilities clipped to `[1e-7, 1-1e-7]`;
- AUROC;
- calibration intercept/slope as diagnostic when estimable.

On the held-out dev calibration-evaluation pool, report metrics twice for noisy conditions:

1. against **observed noisy labels** (`ECE_obs`);
2. against **reference labels** (`ECE_ref`).

On the test pool, report only reference-label metrics.

## Primary estimand

For each dataset and seed:

`BEX_test = ECE10_ref_test(BILATERAL) - ECE10_ref_test(LABEL_ONLY)`.

This is the **bilateral excess calibration error** attributable to the training-topology channel after holding calibration-label noise fixed.

Supporting estimand:

`BEX_Brier = Brier_ref_test(BILATERAL) - Brier_ref_test(LABEL_ONLY)`.

Also report, but do not gate on, the label-noise gap:

`LGAP_test = ECE10_ref_test(LABEL_ONLY) - ECE10_ref_test(CLEAN)`.

## Bootstrap

Use 2,000 paired nonparametric bootstrap resamples of test **positive/negative groups** with RNG seed `20260908`.

For each resample, compute BEX_test separately for the three downstream seeds and average across seeds. Report percentile 95% CI of this mean bilateral excess.

## Frozen gates

### Gate 1: BENCHMARK / MODEL FEASIBILITY

Each dataset must satisfy all:

- public pinned data load successfully;
- generator pairwise accuracy >= 0.65;
- >=98% of target false additions generated for train, dev-fit, and dev-eval;
- clean ComplEx test AUROC >= 0.70 for every downstream seed;
- CLEAN held-out dev ECE_obs <= 0.05 for every downstream seed.

Failure -> **KILL-BENCHMARK**.

Do not change datasets, model, epochs, error mechanism, or calibration method to rescue.

### Gate 2: OBSERVED-CALIBRATION FEASIBILITY

For every dataset and seed, both LABEL_ONLY and BILATERAL must achieve:

`ECE_obs <= 0.05`

on the held-out noisy dev calibration-evaluation pool.

Failure -> **KILL-CALIBRATION-FEASIBILITY**.

This gate is necessary for the intended phenomenon: the system must look calibrated to the constructed graph before we ask whether it is calibrated to the reference graph.

### Gate 3: BILATERAL PHENOMENON

On **both datasets**, all must hold:

1. mean `BEX_test` across three seeds >= **0.03** absolute ECE;
2. `BEX_test > 0` for all three seeds;
3. paired-bootstrap 95% lower bound for mean `BEX_test` > **0.01**;
4. mean `BEX_Brier` >= **0.005**;
5. `BEX_Brier > 0` for all three seeds.

Failure on either dataset -> **KILL-PHENOMENON**.

No new corruption rate, KGE, dataset, or calibrator may be added.

### Gate 4: CALIBRATION SPECIFICITY

If Gate 3 passes, compute:

`AUROC_drop = AUROC_test(LABEL_ONLY) - AUROC_test(BILATERAL)`

averaged across seeds.

Require `AUROC_drop <= 0.05` on both datasets.

If the calibration effect passes but discrimination degrades by more than 0.05 AUROC on either dataset -> **STOP-SPECIFICITY**. The result is treated as broad predictive degradation under noisy topology, not a clean calibration-specific opportunity.

Otherwise -> **CONTINUE**.

## Allowed diagnostics only

- per-relation BEX;
- calibration reliability tables;
- ECE using 15 equal-frequency bins;
- score-distribution shifts;
- fraction and relation distribution of generated false additions;
- overlap of false additions with high-degree entities;
- clean versus bilateral training loss curves.

Diagnostics cannot change the decision.

## Compute bound

- two datasets only;
- one frozen TransE generator per dataset;
- one downstream architecture;
- three downstream seeds;
- two downstream training graphs per seed (clean and bilateral; LABEL_ONLY reuses clean checkpoint);
- 5% false-addition rate only;
- 2,000 bootstrap resamples;
- no hyperparameter search;
- no GPU requirement.

## Interpretation

Possible final decisions:

- `KILL-BENCHMARK`
- `KILL-CALIBRATION-FEASIBILITY`
- `KILL-PHENOMENON`
- `STOP-SPECIFICITY`
- `CONTINUE`

A `CONTINUE` result is only evidence that the phenomenon deserves a stronger validation study. It is **not** sufficient for a paper claim because the pilot reference graph is incomplete and the construction errors are model-produced benchmark corruptions rather than errors from a real extraction pipeline.
