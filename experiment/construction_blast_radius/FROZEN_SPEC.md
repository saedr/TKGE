# Frozen Pilot Specification: Construction Blast Radius

**Frozen:** 2026-09-08, before inspecting any per-error damage outcomes  
**Branch:** `construction-blast-radius-pilot`  
**Status:** falsification pilot only; no learned method is permitted in this phase.

## 1. Scientific question

Do realistic false entity merges have a sufficiently concentrated distribution of *downstream graph damage* that a construction system should distinguish error probability from error consequence?

The object of interest is the consequence term

\[
D(c,G)=\text{downstream damage if construction decision }c\text{ is wrong},
\]

which would eventually support a prospective risk score such as

\[
R(c)=P(c\text{ wrong})\,D(c,G).
\]

This pilot tests only whether the empirical consequence phenomenon is strong enough to justify a method. It does **not** train a predictor for `D`, optimize `R`, or evaluate a new verification policy.

## 2. Frozen scope

### Construction error family

False cross-KG entity merges only.

No random graph corruption, synthetic false merges, missing-edge corruption, or false extracted triples will be added to rescue the pilot.

### Datasets

Exactly two public RealEA 15K datasets from Leone et al. (PVLDB 2022):

1. `DB-YG-15K` — **primary**
2. `DB-WD-15K` — **prespecified replication**

No additional graph may be introduced after outcomes are observed.

### Error generator

Use the published **Paris+** protocol from `epfl-dlab/entity-matchers` and the PARIS implementation distributed by `dig-team/PARIS`.

For each dataset:

- use `721_5folds`;
- sort fold directory names lexicographically and use the first fold only;
- inject that fold's Paris+ training links exactly as the published RealEA implementation does;
- take the final PARIS instance-equivalence output;
- remove training links;
- deduplicate predicted entity pairs;
- define a predicted pair as a false merge iff it is absent from the dataset's complete gold entity alignment.

The third PARIS output field is retained as confidence when it is parseable and finite. If it is unavailable, the confidence diagnostic is recorded as unavailable and is **not** replaced by another matcher or score.

## 3. Clean integrated graph

For each RealEA dataset:

1. Read the two relation-triple graphs.
2. Namespace every relation by source KG (`kg1::r`, `kg2::r`). No relation/schema alignment is inferred.
3. Canonicalize every gold-aligned entity pair into one node. Unmatched entities remain separate.
4. Union the two canonicalized relation graphs.
5. Add a formal inverse label for every relation and inverse edge for every relation triple.
6. Remove exact duplicate labeled edges. Self-loops are retained if present in the clean graph.

This canonicalized graph is the clean reference graph `G`.

A false Paris+ pair maps to two distinct canonical nodes `u` and `v`. Its counterfactual corrupted graph `G_e` is produced by contracting only `u` and `v` into one node, with all of their incident labeled edges inherited. No other predicted merges are inserted simultaneously.

## 4. Primary downstream workload and damage

Define the exhaustive relation-labeled two-hop query-answer set

\[
Q_2(G)=\{(s,r_1,r_2,t): \exists m,\ (s,r_1,m)\in G \land (m,r_2,t)\in G\}.
\]

For false merge `e=(u,v)`, exclude query anchors and answers involving either touched canonical entity. The **primary damage** is

\[
D(e)=|Q_2^{\neg\{u,v\}}(G_e)\setminus Q_2^{\neg\{u,v\}}(G)|.
\]

Thus `D(e)` counts **new false relation-labeled two-hop query answers between untouched entities** caused by one wrong merge. It deliberately excludes answers whose anchor or answer is one of the two merged entities, avoiding endpoint-identity ambiguity after contraction.

Implementation may compute this exact set difference locally rather than materializing every query answer globally; approximation is not allowed for the primary outcome.

## 5. Prespecified descriptive diagnostics

For every false merge, record before intervention:

- Paris confidence, if available;
- labeled incident degree of each endpoint;
- degree sum;
- smoothed degree product `(d(u)+1)(d(v)+1)`;
- PageRank sum on the clean inverse-augmented graph (damping 0.85);
- radius-2 unique-node neighborhood-size sum;
- primary damage `D(e)`.

These are diagnostics/baselines, not learned features for a new model.

## 6. Primary concentration statistic

For `n` false merges with damages sorted descending, let `k=ceil(0.20 n)` and

\[
C_{20}=\frac{\sum_{i=1}^{k} D_{(i)}}{\sum_{i=1}^{n}D_i}.
\]

If total damage is zero, define `C20=0`.

Uncertainty: nonparametric bootstrap over false-merge decisions, 2,000 resamples, RNG seed `20260908`, percentile 95% interval.

Also report the largest single-error share of total damage.

## 7. Frozen gates

### 7.1 Benchmark feasibility gate

For **each** dataset, all of the following must hold:

- the public RealEA files can be retrieved reproducibly without private credentials or manual data substitution;
- Paris+ can execute under the bounded runner;
- at least **100** distinct false predicted merges remain after gold filtering.

If any condition fails, decision = **KILL-BENCHMARK**. Do not switch to a different matcher, dataset, or corruption process inside this pilot.

### 7.2 Concentration gate

The empirical phenomenon passes only if **both datasets** satisfy all of:

1. `C20 >= 0.60`;
2. bootstrap 95% lower bound for `C20 >= 0.50`;
3. no single false merge contributes more than `25%` of total damage;
4. total primary damage is positive.

If any condition fails, decision = **KILL-PHENOMENON**. No method is built.

### 7.3 Simple-explanation / method-opportunity gate

At a 20% verification budget, define damage capture for ranking `h` as

\[
Capture_{20}(h)=\frac{\sum_{e\in Top_{20\%}(h)}D(e)}{\sum_e D(e)}.
\]

The oracle ranking by true `D(e)` has `Capture20 = C20`.

Prespecified non-oracle rankings:

- random (1,000 random rankings; seed sequence derived from `20260908`);
- Paris uncertainty `1-confidence`, if available;
- degree sum;
- degree product;
- PageRank sum;
- radius-2 neighborhood-size sum.

For each dataset, compute `best_simple_ratio = max_h Capture20(h) / C20`, excluding random from the maximum but reporting it.

A learned blast-radius method is allowed in a later phase only if the concentration gate passes **and** `best_simple_ratio < 0.90` on **both** datasets.

If concentration passes but a simple ranking reaches at least 90% of oracle damage capture on either graph, decision = **STOP-METHOD**: the phenomenon may be real, but this pilot does not support a new learned method.

No threshold will be relaxed after seeing outcomes.

## 8. Secondary analyses allowed in this pilot

Only the following prespecified secondary summaries are allowed:

- Spearman correlation between `D` and each available scalar baseline;
- damage quantiles and Lorenz-style cumulative concentration table;
- number and fraction of zero-damage false merges;
- overlap among top-20% sets for oracle and each baseline;
- Paris precision/recall/F1 for the selected fold as a provenance sanity check.

They cannot override a failed primary gate.

## 9. Compute and engineering bounds

- CPU only; no KGE training and no GPU requirement.
- Exactly two 15K graphs and one prespecified Paris+ fold per graph.
- Exact single-error interventions only.
- GitHub-hosted runner target: at most 6 GB Java heap for PARIS and bounded Python memory.
- If PARIS cannot run within the bound, record technical infeasibility rather than silently changing the error generator.
- Fixed bootstrap count: 2,000.
- Results are written as CSV/JSON/Markdown artifacts.
- `main` remains untouched; all work stays on this branch and a draft PR.

## 10. Interpretation policy

- **KILL-BENCHMARK**: public/reproducible benchmark or realistic error source is not usable under the frozen bounds.
- **KILL-PHENOMENON**: false-merge damage is not sufficiently concentrated and stable.
- **STOP-METHOD**: damage concentrates, but a trivial precommit signal nearly recovers the oracle prioritization.
- **CONTINUE**: both concentration and method-opportunity gates pass. Only then may a subsequent, separately frozen phase build a prospective damage predictor / expected-damage verification policy.

A failure cannot be rescued by adding models, datasets, mechanisms, corruption types, query workloads, or post-hoc thresholds.