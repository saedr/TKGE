# Prospective revision feasibility gate — corrected at-risk cohort

## Decision: **GO**

The first automated gate returned `NO-GO` because it counted `Deprecate` events for triples that were **added after the January 1 snapshot** when evaluating snapshot-membership quality. Those triples were not present at prediction time and therefore cannot belong to the prospective at-risk population. Excluding them is required by the study definition (the triple must already exist at time `t`), not a relaxation of the gate.

EMERGE provides a near-term horizon: five progressively larger windows ending about 35 days after each January 1 snapshot. This is not a one-year survival target.

## Corrected cohort health

| Year | Verified Exists | Verified Deprecate | Deprecate prevalence | Deprecate snapshot membership after at-risk filter | Positive relations | Top relation share |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 25,761 | 156 | 0.0060 | 0.981 | 54 | 0.103 |
| 2025 | 24,911 | 147 | 0.0059 | 0.987 | 50 | 0.163 |

At-risk exclusions:

- 2024: 14 of 173 raw `Deprecate` triples were added after Jan 1 and are ineligible; 156 of the 159 remaining `Deprecate` triples are verified in the snapshot (98.1%).
- 2025: 18 of 167 raw `Deprecate` triples were added after Jan 1 and are ineligible; 147 of the 149 remaining `Deprecate` triples are verified in the snapshot (98.7%).

## Controls-only 2024 → 2025 baseline

| Baseline | 2025 AUPRC | AUROC | Brier |
|---|---:|---:|---:|
| structural + history | 0.020649 | 0.739094 | 0.005801 |
| relation only | 0.052316 | 0.801126 | 0.005663 |
| full controls | 0.058969 | 0.821049 | 0.005675 |
| prevalence | 0.005866 | — | — |

The full controls baseline is strong relative to prevalence. A KGE therefore has to add information beyond relation identity, degree/frequency, age, and past relation invalidation rate.

## Feasibility conclusion

- Positive count is sufficient: 156 verified Deprecates in 2024 and 147 in 2025.
- Positive relations are diverse: 54 in 2024 and 50 in 2025; no single relation dominates.
- Snapshot alignment is acceptable after enforcing the correct at-risk definition.
- A clean out-of-time controls baseline is executable.
- Future passages, removal dates, deprecation reasons, qualifiers, and future text are not used as features.

## Next scientific gate

Use one fixed KGE architecture, fit it separately on the Jan-1 2024 and Jan-1 2025 snapshots with identical hyperparameters, relation-normalize its triple score within each snapshot, add that score to the frozen controls model, and test the incremental 2025 AUPRC.

The continuation threshold remains:

`ΔAUPRC >= 0.005` with a paired-bootstrap 95% CI excluding zero.

Do not add architectures or features if that gate fails.
