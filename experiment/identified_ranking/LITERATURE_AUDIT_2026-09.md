# Literature Audit — Partial Identification of KGE Rankings

Date: 2026-09-08

## Verdict

QUALIFIED PASS.

The broad claims are not novel:

- possible-world semantics for uncertain/probabilistic databases and graphs are established;
- top-k/rank aggregation over possible worlds is established in probabilistic databases;
- uncertain KGE models already predict confidence scores for uncertain facts;
- conformal prediction for deterministic KGE answer sets is established;
- conformal prediction intervals for uncertain-KG fact confidence are established (UnKGCP, EMNLP 2025);
- predictive multiplicity of KGE rankings across random seeds is established;
- adversarial/sensitivity analyses of KGE predictions under graph modifications are established;
- credal and distributionally robust graph learning exist outside this exact KGE setting.

The surviving claim is narrower:

> Statistically valid uncertainty about individual KG facts may leave the identity of a downstream learned KGE answer only partially identified. In particular, fact-level prediction intervals can induce a set of admissible graph completions under which the downstream top-1 KGE answer changes, even when the downstream model, initialization, and query are held fixed.

The object of interest is therefore not a point probability, a fact-level prediction interval, a conformal answer set, or random-seed multiplicity. It is a downstream *decision identification set* induced by uncertainty in the graph itself.

## Closest prior work and boundary of the claim

### Uncertain KGE

Chen et al., AAAI 2019, "Embedding Uncertain Knowledge Graphs" introduced UKGE and the CN15k/NL27k/PPI5k benchmarks. Uncertain facts carry confidence scores; UKGE predicts confidence for unseen triples. This establishes uncertain KGE but not downstream rank identification over admissible graph completions.

https://ojs.aaai.org/index.php/AAAI/article/view/4210

### Fact-level uncertainty intervals

Zhu et al., EMNLP 2025, "Certainty in Uncertainty: Reasoning over Uncertain Knowledge Graphs with Statistical Guarantees" introduced UnKGCP. It constructs conformal prediction intervals for the true confidence score of unseen uncertain-KG facts. At 90% nominal coverage, the paper reports empirical coverage near the target on CN15k, NL27k, and PPI5k across UKGE/PASSLEAF/BEUrRE. It does not propagate those fact-level intervals into uncertainty about a separately learned downstream ranking.

https://aclanthology.org/2025.emnlp-main.441/
https://github.com/0sidewalkenforcer0/UnKGCP

Pinned upstream implementation for the pilot:
6cdf658be1825be0b369084859dff129ae3c113f

### Conformal KGE answer sets

Zhu et al., NAACL 2025, "Conformalized Answer Set Prediction for Knowledge Graph Embedding" constructs answer sets with coverage guarantees for deterministic-KG link prediction. This quantifies uncertainty in answers directly, conditional on one observed KG. It does not ask whether the answer changes across graph completions compatible with uncertain facts.

https://aclanthology.org/2025.naacl-long.32/

### Predictive multiplicity

Zhu et al., Findings of EMNLP 2024, "Predictive Multiplicity of Knowledge Graph Embeddings in Link Prediction" shows that similarly performing KGE models trained on the same graph can disagree on 8–39% of test queries. Their source of variation is model/training multiplicity. The proposed phenomenon instead holds the downstream training seed fixed and varies only the admissible graph completion; random-seed multiplicity is an explicit control.

https://aclanthology.org/2024.findings-emnlp.19/

### Possible-world ranking in probabilistic databases

Possible-world semantics and top-k/rank queries over probabilistic databases predate modern KGE. Work includes UTop-k/URank-k/global top-k and consensus-answer formulations. These methods aggregate deterministic query answers or tuple scores over probabilistic worlds; they do not retrain a learned KGE as a function of the admissible graph and then ask whether the learned link-prediction decision is identified.

Representative sources:
- Soliman/Ilyas-style probabilistic top-k literature
- Li & Deshpande, "Consensus Answers for Queries over Probabilistic Databases"
- Olteanu et al., "Ranking Query Answers in Probabilistic Databases"

### Probabilistic KG query answering

Gaur et al., "Computing and Maintaining Provenance of Query Result Probabilities in Uncertain Knowledge Graphs" uses possible-world semantics for conjunctive KG queries and probabilistic inference. This is fixed-query probabilistic semantics, not uncertainty propagation through learned KGE training.

https://arxiv.org/abs/2108.07758

### Adversarial KGE robustness

Pezeshkpour et al., NAACL 2019, "Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications" identifies graph additions/removals that change a target link prediction after retraining. This is a crucial predecessor: graph changes can change KGE predictions. However, its perturbations are optimized adversarially and are not derived from statistically valid uncertainty intervals around uncertain facts.

https://aclanthology.org/N19-1337/

Recent robustness/certification work also studies embedding- or graph-perturbation stability. Therefore the project must not claim that KGE predictions can change under graph perturbations as novel.

### Credal / imprecise graph learning

Credal learning, imprecise probability, and distributionally robust graph learning exist, including recent credal GNN work for node classification. The project must not claim credal graph prediction generally as new.

## Exact novelty claim allowed if the phenomenon passes

A defensible claim would be:

> We study whether statistically valid fact-level uncertainty leaves a downstream KGE ranking identified. We define empirical rank non-identification with respect to graph completions induced by conformal confidence intervals, separate it from ordinary KGE predictive multiplicity, and measure how often nominally decisive link predictions admit conflicting top-ranked answers under admissible graph completions.

Do not claim:

- first uncertain KGE;
- first possible-world KG reasoning;
- first KGE uncertainty interval;
- first conformal KGE uncertainty;
- first KGE ranking instability;
- first credal graph model;
- first robust KGE.

## Dataset audit

### NL27k — primary

Derived from NELL, a large-scale knowledge base built by semi-automatic extraction. Standard benchmark size is about 27k entities, 404 relations, and 175k uncertain relation facts. This is the strongest primary dataset because the uncertainty is directly tied to an automatic knowledge-acquisition process.

### CN15k — prespecified replication

Derived from ConceptNet, with about 15k entities, 36 relations, and 241k uncertain facts. It is useful as a qualitatively different uncertainty regime. UnKGCP reports substantially wider intervals on CN15k than NL27k for the UKGE backbone, making it a meaningful replication rather than a near-duplicate dataset.

### PPI5k — not in the first pilot

PPI5k is biologically valuable, but its STRING scores have a different evidential interpretation and the graph is very dense with only seven relation types. It is reserved for later external validation only if the phenomenon survives both prespecified datasets.

## Scientific risk

The main risk is that conformal fact intervals are so wide that almost every candidate fact is ambiguous, making downstream non-identification trivial. The frozen pilot therefore has an informativeness gate on the fraction of interval-threshold-crossing facts. A second risk is ordinary KGE random-seed multiplicity; the pilot therefore compares graph-induced disagreement against a same-graph seed control.

## Audit conclusion

Proceed to a phenomenon-only falsification. No new KGE architecture, Bayesian model, credal optimizer, value-of-information method, or robust decision rule is justified unless the frozen phenomenon gate passes on both NL27k and CN15k.