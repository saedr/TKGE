# Prospective KGE reliability pilot — DistMult only

## Decision: **KILL**

Frozen scientific gate: add the relation-normalized DistMult signal to the full controls-only model, then require **ΔAUPRC ≥ 0.005** on the untouched 2025 cohort and a paired-bootstrap 95% CI for ΔAUPRC entirely above zero.

## 2025 primary result

| Quantity | Value |
|---|---:|
| Eligible triples | 25,058 |
| Deprecate positives | 147 |
| Positive prevalence | 0.005866 |
| Controls AUPRC | 0.058969 |
| Controls + DistMult AUPRC | 0.058658 |
| **ΔAUPRC** | **-0.000311** |
| Paired-bootstrap 95% CI for Δ | [-0.003325, 0.000606] |
| Controls AUROC | 0.821049 |
| Controls + DistMult AUROC | 0.820162 |

The frozen gate fails: the point estimate is negative, is far below the required +0.005 improvement, and the confidence interval includes zero.

## Training sanity

| Snapshot | Triples | Entities | Relations | First epoch loss | Final epoch loss | Positive > random-negative rate | Sanity |
|---|---:|---:|---:|---:|---:|---:|---|
| 2024 | 36,306,231 | 5,977,180 | 1,297 | 1.110296 | 0.656449 | 0.9841 | Pass |
| 2025 | 37,542,677 | 6,100,962 | 1,351 | 1.103624 | 0.643240 | 0.9845 | Pass |

The null incremental result is therefore not explained by an obvious DistMult training failure.

## Frozen KGE configuration

- DistMult only.
- 32-dimensional embeddings.
- Two full-snapshot epochs per year.
- One uniformly sampled tail negative per positive.
- Sparse Adagrad, learning rate 0.10.
- Identical settings for 2024 and 2025.
- Relation normalization uses the score distribution of all triples of that relation in the Jan-1 snapshot, never the future-labeled cohort.
- The risk set includes only triples verified to exist in the corresponding Jan-1 snapshot.
- The controls model is unchanged from the feasibility gate: relation identity, head/tail degree, relation frequency, triple age/missingness, and historical relation deprecation rate.
- Bootstrap resamples 2025 test triples with paired baseline/augmented predictions; models are not refit inside each bootstrap replicate.

## Frozen interpretation

The prospective DistMult score does not add meaningful information about near-term EMERGE deprecation beyond simple structural and historical controls. Per the prespecified kill gate, do not add ComplEx, TransE, GNNs, extra targets, or hand-tuned features to rescue this idea.

Workflow run: https://github.com/saedr/TKGE/actions/runs/34007765134
