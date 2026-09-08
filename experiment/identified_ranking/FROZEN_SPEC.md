# Frozen Falsification Specification — Identification of KGE Rankings under Fact Uncertainty

Frozen: 2026-09-08, before implementation or outcome inspection.

## Scientific question

Do statistically valid fact-level uncertainty intervals leave downstream KGE top-1 decisions non-identified across admissible graph completions, beyond ordinary random-seed predictive multiplicity?

This is a phenomenon test only. No new method is permitted unless the continuation gate passes on both prespecified datasets.

## Data

Exactly two datasets:

1. NL27k — primary.
2. CN15k — replication.

Use the official splits distributed with the UnKGCP/UKGE benchmark family. No dataset substitution is allowed.

Pinned UnKGCP upstream implementation:
`0sidewalkenforcer0/UnKGCP@6cdf658be1825be0b369084859dff129ae3c113f`

## Fact-level uncertainty

Use the UKGE backbone and the adaptive UnKGCP conformal interval construction from the pinned upstream implementation.

Nominal confidence level: 0.90.

Use the official train split to fit the UnKGE backbone and the official validation/calibration split for conformal calibration. The test split is never used to fit the confidence model or interval procedure.

The strong-fact threshold is fixed at `tau = 0.85` for both NL27k and CN15k, matching the original UKGE relation-fact classification protocol.

For each held-out uncertainty-pool fact j with conformal interval `[L_j, U_j]`:

- definitely strong: `L_j > tau`;
- definitely weak: `U_j <= tau`;
- ambiguous: `L_j <= tau < U_j`.

The interval point prediction is used only to define the nominal graph: an ambiguous fact enters the nominal graph if its UKGE point prediction exceeds tau.

## Test split partition

Sort official test triples lexicographically by `(relation, head, tail)` and assign alternating rows:

- even-indexed rows -> uncertainty pool U;
- odd-indexed rows -> downstream query pool Q.

True test confidence scores are used only for diagnostics/coverage and for selecting strong evaluation queries after intervals are constructed. They are not used to decide inclusion of U facts in any admissible world.

## Benchmark-validity gates

A dataset is scientifically usable only if all are satisfied:

1. empirical 90% interval coverage on U is at least 0.88;
2. at least 100 U facts are ambiguous;
3. ambiguous fraction among U is between 0.05 and 0.80 inclusive;
4. at least 500 Q facts have true confidence > tau and can be used as strong downstream tail queries.

Failure of any condition on either dataset -> `KILL-BENCHMARK`.

The upper ambiguous-fraction gate prevents a trivial result in which almost every fact is admissible in either state.

## Base and admissible graphs

Construct a deterministic strong-fact base graph from official training facts whose observed confidence is > tau.

Add all definitely-strong U facts to every admissible graph.

Definitely-weak U facts are never added.

Ambiguous U facts define the graph-completion ambiguity set: each may be either included or excluded.

The full ambiguity set is exponential. The falsification uses a prespecified witness search, not an estimate of a posterior probability.

Evaluate exactly 12 admissible graph completions:

1. LOWER: exclude every ambiguous fact;
2. UPPER: include every ambiguous fact;
3. NOMINAL: include an ambiguous fact iff the UKGE point prediction > tau;
4. nine deterministic random witness worlds generated independently with Bernoulli(0.5) inclusion for each ambiguous fact, RNG seed 20260908.

These random worlds are search points in the admissible set. Their frequencies have no probabilistic interpretation.

## Downstream KGE

Use ComplEx only.

Fixed hyperparameters:

- embedding dimension: 64;
- epochs: 10;
- batch size: 4096;
- negatives per positive: 1;
- optimizer: Adam;
- learning rate: 0.001;
- training seed for all 12 graph worlds: 11.

Use the same initial parameter tensors for corresponding entities/relations across all graph worlds and deterministic data-order seeding where feasible, so the world comparison isolates graph-completion variation.

No hyperparameter tuning is allowed.

## Downstream query set

From Q, retain facts with true confidence > tau.

Take the first 500 after the same lexicographic ordering. Evaluate tail prediction `(h, r, ?)`.

Candidates are all entities. Use filtered ranking against all known strong facts in train + validation + test (confidence > tau) only for correctness diagnostics. The primary top-1 identity comparison is the raw model winner and does not use ground-truth filtering to force agreement.

## Random-seed multiplicity control

On the NOMINAL graph only, train ComplEx with exactly five seeds:

`11, 23, 37, 53, 71`.

Use identical hyperparameters otherwise.

This control estimates ordinary predictive multiplicity on one fixed graph.

## Primary quantities

For each query q:

### Empirical world non-identification witness

`NI_world(q) = 1` iff at least two of the 12 admissible graph worlds produce different top-1 entities.

`WitnessRate = mean_q NI_world(q)`.

This is a lower bound on non-identification over the full ambiguity set because only 12 admissible worlds are searched.

### Pairwise world disagreement

Average, over all unordered pairs of admissible worlds, of the fraction of queries whose top-1 entities differ.

Call this `WorldPairDisagree`.

### Pairwise seed disagreement

Average, over all unordered pairs of the five NOMINAL-graph seed runs, of the fraction of queries whose top-1 entities differ.

Call this `SeedPairDisagree`.

### Excess graph-induced disagreement

`ExcessDisagree = WorldPairDisagree - SeedPairDisagree`.

## Conventional decisiveness diagnostic

On the NOMINAL graph, seed 11, compute the score margin between the top-1 and top-2 entity for each query.

Define HIGH-MARGIN as the top quartile of these margins within each dataset.

Compute `HighMarginWitnessRate`, the world non-identification witness rate restricted to HIGH-MARGIN queries.

This is descriptive evidence that nominally decisive predictions can remain non-identified. The quartile is fixed before seeing outcomes.

## Secondary diagnostics

Report:

- interval coverage on U;
- average interval length;
- ambiguous count/fraction;
- WitnessRate;
- WorldPairDisagree;
- SeedPairDisagree;
- ExcessDisagree;
- HighMarginWitnessRate;
- fraction of queries whose correctness changes across worlds;
- nominal MRR/Hits@10 on the 500 strong queries;
- top-1 entropy across the 12 worlds (frequency used descriptively only, not as posterior probability);
- relation-level witness rates where at least 20 evaluation queries exist.

## Continuation gate

A dataset passes the phenomenon gate only if ALL are satisfied:

1. `WitnessRate >= 0.25`;
2. `WorldPairDisagree >= 0.10`;
3. `ExcessDisagree >= 0.05`;
4. `HighMarginWitnessRate >= 0.10`;
5. nominal seed-11 Hits@10 on the strong-query set is at least 0.20, preventing interpretation of arbitrary rankings from a failed downstream model.

Both NL27k and CN15k must pass.

If either dataset fails -> `KILL-PHENOMENON`.

If both pass -> `CONTINUE` to a second-stage project on computing or approximating downstream identification sets. No method choice is frozen yet.

## Interpretation constraints

A passing pilot would establish only that fact-level uncertainty can propagate into downstream rank non-identification under this benchmark construction.

It would NOT establish:

- calibrated posterior probabilities over graph worlds;
- sharp identification regions over the full exponential ambiguity set;
- real-world clinical or operational risk;
- superiority of any new uncertainty method.

A failed pilot must not be rescued by:

- changing confidence level;
- changing tau;
- adding PPI5k;
- changing KGE model;
- increasing epochs/dimension;
- changing the number or construction of witness worlds;
- weakening thresholds;
- redefining the primary metric.

## Compute bound

CPU or single-GPU execution only. Exactly two datasets, one UnKGCP/UKGE backbone, one downstream KGE, 12 graph worlds plus five nominal seed controls per dataset. No model sweep.