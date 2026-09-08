# Construction blast-radius falsification result

Status: **KILL-PHENOMENON**

This file records the first scientifically valid execution of the frozen pilot. The scientific specification and thresholds were committed before these results were observed. The earlier KILL-BENCHMARK workflow outcomes were archive-layout engineering failures and are not scientific results.

Valid GitHub Actions run: 34276824209
Head commit used for the run: `37a9e9ec73a9864d8b04b0ec5235f3c69aaaa6b2`

## Frozen gate

Continue only if both datasets satisfy all of:

- at least 100 model-produced false merges;
- C20 >= 0.60;
- 95% bootstrap lower bound for C20 >= 0.50;
- no single false merge accounts for more than 25% of total damage.

If the concentration gate passes, method development is still stopped if any prespecified simple structural baseline recovers at least 90% of oracle top-20%-budget damage capture on either graph.

## Results

| Dataset | False merges | C20 | 95% bootstrap CI | Max single share | Best simple / oracle | Decision contribution |
|---|---:|---:|---:|---:|---:|---|
| DB-YG-15K | 1,320 | 0.8685 | [0.7717, 0.9206] | 0.2860 | 0.9839 | fails max-single-share gate |
| DB-WD-15K | 336 | 0.9979 | [0.9683, 0.9993] | 0.8479 | 0.9995 | fails max-single-share gate |

Overall decision: **KILL-PHENOMENON**.

The concentration itself is extremely strong, but the preregistered non-degeneracy condition fails on both datasets. The effect is especially pathological on DB-WD-15K, where one wrong merge produces 84.8% of all measured downstream damage.

## Prespecified baselines

### DB-YG-15K

- Oracle C20: 0.8685
- degree product capture20: 0.8545
- degree sum capture20: 0.8515
- PageRank sum capture20: 0.8435
- radius-2 sum capture20: 0.5478
- PARIS uncertainty capture20: 0.0521
- best simple/oracle ratio: 0.9839

### DB-WD-15K

- Oracle C20: 0.9979
- degree product capture20: 0.9970
- degree sum capture20: 0.9972
- PageRank sum capture20: 0.9974
- radius-2 sum capture20: 0.9753
- PARIS uncertainty capture20: 0.0006
- best simple/oracle ratio: 0.9995

Thus even if the max-single-share condition had passed, the frozen STOP-METHOD criterion would have prevented learning a blast-radius estimator: simple graph structure already captures essentially all oracle damage concentration.

## Dominant observed errors

DB-YG-15K top error:

- `http://dbpedia.org/resource/Scotland_national_football_team` -> `Scotland`
- damage: 57,908 / 202,458 = 28.6%
- PARIS confidence: approximately 1.0

DB-WD-15K top error:

- `http://dbpedia.org/resource/Film_director` -> `http://www.wikidata.org/entity/Q5`
- damage: 1,142,128 / 1,346,958 = 84.8%
- PARIS confidence: 1.0

## Interpretation

The pilot does support two descriptive observations: real matcher-produced false merges have radically unequal downstream consequences, and probability-of-error alone can rank the most damaging errors very poorly. However, under the frozen scientific gate, this is not sufficient to justify a new learned method or a top-tier paper around prospective blast-radius prediction. The measured consequence is dominated by obvious high-connectivity hub errors, and degree/PageRank baselines nearly saturate the oracle.

Per the preregistration, do not add datasets, matchers, damage definitions, mechanisms, or learned models to rescue this direction. `main` remains untouched; this result stays on the isolated draft branch/PR.
