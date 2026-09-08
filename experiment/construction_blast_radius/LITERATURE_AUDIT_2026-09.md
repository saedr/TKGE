# Literature Audit: Prospective Construction Blast Radius

**Search horizon:** through 2026-09-08  
**Decision:** **qualified proceed** to a falsification pilot.

## Proposed object

For a candidate KG construction decision `c` (especially an entity merge), distinguish:

- `P(c wrong)`: probability the construction decision is incorrect;
- `D(c,G)`: counterfactual downstream damage if that specific decision is wrong;
- eventual risk `R(c)=P(c wrong) D(c,G)`.

The proposed contribution is **not** merely to be risk-aware, downstream-aware, query-aware, or impact-aware. All of those ideas have prior art. The candidate gap is a **pre-commit, per-construction-decision consequence variable** measured against downstream graph workloads and used to prioritize verification by damage prevented.

## Closest prior-art families

| Area | Representative prior work | What is already established | What remains different from this project |
|---|---|---|---|
| Decision-theoretic ER | Dey, Sarkar & De, *Management Science* 1998 | Entity matching under uncertainty can minimize explicit match-classification costs. | Costs are decision/classification costs, not graph-dependent downstream consequence of an individual wrong merge. |
| Risk-aware ER / human verification | Chen et al., RiskER (2018); r-HUMO line | A machine-labeled pair can be assigned a risk of being mislabeled and high-risk cases sent to humans. | Their risk is primarily likelihood of label error; no separate per-error downstream damage term. |
| Crowd/active ER question selection | Whang, Lofgren & Garcia-Molina, PVLDB 2013; Yalavarthi et al. 2017 | Questions can be selected by expected improvement in the **ER clustering result**, not uncertainty alone. | The impact target is intrinsic ER/clustering accuracy, not the behavior of a downstream graph/query/KGE workload if a merge is accepted incorrectly. |
| Active entity alignment | ActiveEA, EMNLP 2021 | Annotation value can mix uncertainty with graph-structural influence on neighboring alignment uncertainty. | Structural influence measures usefulness of a label to the EA learner, not damage caused by accepting a false construction decision. |
| Query-driven ER | QDA (PVLDB 2013 / TKDE 2016); BrewER / progressive query-driven ER | ER can be restricted/prioritized to records needed to answer a current query. | The goal is avoiding unnecessary ER work for a query, not estimating consequence conditional on a specific false merge over a downstream workload. |
| Downstream-aware cleaning | ActiveClean, PVLDB 2016; ML-aware cleaning literature | Cleaning effort can be allocated according to effect on a downstream statistical/ML model. | Strong conceptual ancestor, but not a graph-construction merge object and not the proposed per-merge graph/query damage variable. |
| Query influence / causality | Database causality/responsibility; influential-tuples work | Individual input tuples can be ranked by effect on query answers/probabilistic query output. | Supplies formal precedent for downstream sensitivity, but does not address uncertain ER/KG construction decisions or false-merge verification. |
| Record linkage + downstream inference | Steorts/Tancredi/Liseo 2018; Kaplan/Betancourt/Steorts 2018/2022; continuing 2026 linkage-inference work | Linkage uncertainty should propagate into later regression/inference; linkage and downstream models may be joint. | Marginalizes/propagates linkage uncertainty for valid inference rather than pricing prospective damage of each candidate merge and prioritizing verification. |
| False-merge network distortion | Fegley & Torvik, PLOS ONE 2013; later author-disambiguation/network-measurement studies | Lumping/merge errors can distort global network statistics nonlinearly, and a small number of lumping operations can account for a large share of changes in some metrics. | This directly removes novelty from the weak claim “merge damage is unequal/concentrated.” The remaining question is whether individual construction errors can be prospectively valued for downstream KG workloads and verification. |
| KG cleaning/repair | KGClean and subsequent KGE-/constraint-/LLM-assisted cleaning systems | KGs can be checked and repaired using embeddings, constraints, provenance, or verification modules; repair propagation is recognized. | Typically optimizes error detection/repair quality itself, not consequence-aware prioritization of a candidate merge before commit. |
| KG construction quality | KG construction surveys; KGCQual (2026); uncertainty-management work | Construction errors and pipeline choices affect downstream KG quality and link prediction; construction quality can be measured intrinsically and extrinsically. | Aggregate/system-level quality rather than a counterfactual `D(c,G)` attached to each candidate decision. |
| Production KG identity practice | *Curate Before You Connect: Identity and Ontology Tagging in a Production Knowledge Graph* (2026) | Identity merges are operationally destructive; merge mistakes can corrupt otherwise correct entities. | Motivates the failure mode but does not define or predict per-merge downstream damage. |
| Current agentic ER | AgenticER (2026) | ER can be framed as sequential decision-making over evidence acquisition, human queries, accuracy, cost, and latency. | Its decision utility is not a downstream KG blast-radius variable for each proposed identity merge. |
| Source-level impact on ER graphs | U.S. Patent 12,596,692 (2026) | A candidate **data source** can be profiled for expected impact on an ER graph (new touchpoints, relationships, potential consolidations). | Important terminology/claim constraint: “expected impact on an ER graph” already exists at source level. It is not per candidate identity decision, nor downstream workload damage conditional on error. |
| Coreference learning-to-search | Clark & Manning, ACL 2015 | Local cluster-merge actions can be assigned rollout costs based on eventual coreference metric. | Intrinsic structured-prediction loss, not prospective KG construction damage on downstream graph consumers. |

## Exact novelty verdict

I did **not** find, through the search horizon, a work that jointly instantiates all four elements below:

1. a **specific candidate KG construction merge** `c`;
2. a counterfactual **downstream graph/query/KGE consequence** `D(c,G)` defined conditional on that merge being wrong;
3. a pre-commit expected-damage framing such as `P(c wrong) × D(c,G)`;
4. selective verification evaluated by **downstream damage prevented** under a review budget.

This is a **qualified novelty claim**, not proof that no obscure paper exists. Several adjacent literatures contain each component separately, so the eventual paper must cite and distinguish them explicitly.

## Claims that are not novel enough

The project must **not** claim as its primary novelty that:

- entity resolution should be risk-aware;
- uncertain merges should receive human review;
- active ER should consider impact as well as uncertainty;
- downstream tasks should influence data cleaning;
- linkage uncertainty affects downstream inference;
- false merges can propagate or distort a network;
- a few merge errors can sometimes dominate global network statistics;
- graph centrality can indicate important entities;
- query-aware cleaning is new;
- “blast radius” as a generic graph concept is new.

## Defensible candidate claim if the pilot succeeds

A defensible formulation is:

> KG construction systems usually estimate whether a candidate decision is wrong, while treating the cost of different wrong decisions as interchangeable. We define a construction decision's **counterfactual downstream consequence** and test whether realistic false entity merges exhibit stable concentration in this consequence. If so, construction-time verification should optimize expected downstream damage prevented rather than error count alone.

A stronger method claim is allowed only after the frozen concentration and simple-baseline gates pass.

## Why the falsification pilot is necessary

The strongest scientific threat is not merely prior art; it is triviality. For a false merge, downstream two-hop contamination may be almost a deterministic function of degree, PageRank, or local neighborhood size. If simple structural heuristics recover nearly all of the oracle damage ranking, a learned blast-radius method would offer little research value.

The second threat is diffuseness. If realistic model-produced false merges do not show strong concentration on two independent KG pairs, then expected-damage prioritization has little leverage.

The third threat is benchmark realism. Random corruption would make the experiment easy to manipulate and has known problems as a stand-in for naturally occurring extraction/disambiguation error. The pilot therefore uses RealEA plus a published entity-alignment system rather than random node pairs.

## Dataset audit

### Selected: RealEA DB-YG-15K and DB-WD-15K

Reasons:

- real multi-relational KG structure from DBpedia paired with YAGO/Wikidata;
- sampled to better reflect realistic degree distributions;
- unlike classic OpenEA benchmarks, RealEA does not assume that every entity has a counterpart;
- complete alignment truth for the sampled graphs;
- published Paris+ protocol yields nonzero false positives, providing model-produced false merge decisions rather than random corruption;
- 15K scale keeps exact single-error counterfactual analysis bounded.

Caveat: the public repository currently flags its dataset link as needing an update. Reproducible data access is therefore part of the **predeclared benchmark-feasibility gate**, not something to work around after seeing results.

### Considered but not selected for this pilot

- **OpenEA 15K:** convenient and KG-native, but its closed 1-to-1 matching assumption is exactly one of the realism problems RealEA was designed to correct.
- **S2AND:** excellent real author-disambiguation labels and realistic model errors; however, it is less KG-native/multi-relational and considerably heavier. Strong future external validation only if the primary phenomenon passes.
- **OAEI KG/instance matching tracks:** attractive real system outputs, but partial or task-specific reference alignments can make “not in gold = false merge” unsafe without careful dataset-specific adjudication.
- **AffilKG / extraction benchmarks:** useful for a later false-edge construction-error family, but they do not provide the same clean package of false entity-merge candidates + complete alignment truth + multi-relational graph for this first probe.
- **NEV-style verification benchmarks:** relevant to construction verification, but current public examples appear too small in actual model errors for a stable concentration estimate.

## Positioning if successful

The paper should be positioned at the intersection of trustworthy KG construction, entity resolution/alignment, selective prediction/verification, and downstream-aware data quality. The expected top-tier story is not a new KGE scoring trick. It is a construction-time decision principle with KGE as one possible downstream consumer after the phenomenon is established.

## References / links used in this audit

- Dey, Sarkar & De. *A Probabilistic Decision Model for Entity Matching in Heterogeneous Databases.* Management Science, 1998. DOI: 10.1287/mnsc.44.10.1379.
- Whang, Lofgren & Garcia-Molina. *Question Selection for Crowd Entity Resolution.* PVLDB, 2013.
- Altwaijry, Kalashnikov & Mehrotra. *Query-Driven Approach to Entity Resolution.* PVLDB 2013; extended TKDE version 2016.
- Zecchini. *Progressive Query-Driven Entity Resolution.* SISAP, 2021.
- Krishnan et al. *ActiveClean: Interactive Data Cleaning for Statistical Modeling.* PVLDB, 2016.
- Chen et al. RiskER / risk-analysis line for human-machine ER, 2018.
- *ActiveEA: Active Learning for Entity Alignment.* EMNLP, 2021.
- Steorts, Tancredi & Liseo. *Generalized Bayesian Record Linkage and Regression with Exact Error Propagation.* 2018.
- Kaplan, Betancourt & Steorts. *A Practical Approach to Proper Inference with Linked Data.* 2018; The American Statistician, 2022.
- Fegley & Torvik. *Has Large-Scale Named-Entity Network Analysis Been Resting on a Flawed Assumption?* PLOS ONE, 2013.
- Leone et al. *A Critical Re-evaluation of Neural Methods for Entity Alignment.* PVLDB, 2022.
- Suchanek, Abiteboul & Senellart. *PARIS: Probabilistic Alignment of Relations, Instances, and Schema.* VLDB, 2012.
- Papadakis et al. *AgenticER: the next frontier in Entity Resolution.* arXiv, 2026.
- U.S. Patent 12,596,692. *Source scoring for entity representation systems.* issued 2026-04-07.
- Current KG construction/uncertainty/quality literature screened through September 2026, including KG construction surveys, uncertainty management, KGCQual, KG cleaning/repair, dynamic verification, and production identity-curation work.
