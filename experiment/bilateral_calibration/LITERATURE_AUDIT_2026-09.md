# Literature audit: KGE calibration under construction uncertainty

Date: 2026-09-08

## Verdict

**QUALIFIED PASS TO FALSIFICATION.**

The broad claim "a KGE can be calibrated to an observed KG but miscalibrated relative to truth" is **not novel**. The surviving question is narrower:

> When structured construction errors enter a KG as observed edges, they can perturb both (i) the topology used to learn KGE representations and (ii) the labels used to fit a post-hoc calibrator. Does this *bilateral construction-noise channel* produce additional calibration failure relative to a matched label-only control that sees the same noisy calibration labels but clean training topology?

No paper found through 2026-09-08 jointly instantiates: KGE/link prediction; non-adversarial or model-produced false KG edges; a clean/reference graph; a matched label-only control; and calibration degradation attributable specifically to the training-topology channel.

This is a falsifiable gap, not yet a paper claim.

## Closest prior work and what it rules out

### KGE calibration

- Tabacof & Costabello, *Probability Calibration for Knowledge Graph Embedding Models* (2019/ICLR workshop): establishes Platt scaling/isotonic calibration for KGE. We cannot claim KGE probability calibration itself is new.
- Safavi, Koutra & Meij, *Evaluating the Calibration of Knowledge Graph Embeddings for Trustworthy Link Prediction* (EMNLP 2020): the most important predecessor. It shows calibration that works under the closed-world assumption deteriorates under open-world human judgments. Therefore we cannot claim novelty for "observed KG labels versus external truth" or for a CWA-to-OWA calibration gap.
- Zhu et al., *A Closer Look at Probability Calibration of Knowledge Graph Embedding Models* (2022): further studies design choices and evaluation of KGE calibration.
- Yang et al., *KGE Calibrator: An Efficient Probability Calibration Method of Knowledge Graph Embedding Models for Trustworthy Link Prediction* (EMNLP 2025): recent KGE-specific post-hoc calibration. It does not study construction-noise contamination of the graph used to learn embeddings.
- Recent conformal KGE/KGC work (NAACL/ACL/EMNLP 2025) gives coverage-oriented uncertainty sets/intervals but assumes the calibration/reference labels are the target truth; it does not isolate graph-construction noise.

### Uncertain/probabilistic KGs

- BEUrRE (2021) and subsequent uncertain-KG embedding work model per-triple confidence/probabilistic semantics. This rules out any claim that uncertain facts or probabilistic KGs are new. Their setup treats uncertainty/confidence as part of the input rather than asking whether a conventional KGE becomes falsely calibrated because an automatically constructed graph is wrong.

### Calibration with noisy labels

- General supervised-learning literature already shows that calibrating with noisy labels can target the wrong posterior and develops calibration methods robust to label noise (including Penso et al. 2024 and TransTS 2026). Therefore "noisy calibration labels cause miscalibration against clean truth" is not a KG contribution.

The matched label-only arm is mandatory. If it explains the effect, this direction is a KG restatement of known noisy-label calibration and must be killed.

### Structural/edge noise in graph learning

- Zhou et al., *Combating Bilateral Edge Noise for Robust Link Prediction* (NeurIPS 2023), is the closest mechanism-level predecessor. It explicitly shows that noisy edges perturb both input topology and target labels and studies accuracy/representation collapse. It does **not** study probability calibration or KGE confidence against a clean reference graph.
- Kapoor et al. (2024) evaluate KGE performance under graph/label/parameter perturbations, but not calibration or a matched topology-vs-label decomposition.
- *The Confidence Trap: Calibration Attacks for Graph Neural Networks* (2026) shows adversarial structural perturbations can attack GNN calibration while preserving labels. This rules out a broad claim that "graph structure can affect calibration." It is a worst-case adversarial GNN/node-classification setting, not non-adversarial KG construction errors or KGE link prediction.

### Noisy KGs / KG error detection

- CPConvKE and related noisy-KGE methods study ranking/robustness under noisy triples.
- Liu, Liu & Hu, *Knowledge Graph Error Detection with Contrastive Confidence Adaption* (AAAI 2024), constructs semantically similar and adversarial false triples for FB15k-237 and WN18RR and studies error detection. Its public generator demonstrates the need for hard, relation-compatible/model-produced errors rather than uniform random corruption. It does not evaluate calibration.
- *Robust Knowledge Graph Embedding via Denoising* (2025/2026) studies robustness/certification under perturbation, not clean-reference probability calibration.

### Automatic KG construction datasets

Recent resources provide genuine extraction errors, but none found gives the combination needed for a cheap controlled KGE calibration experiment: a sufficiently connected multi-relational automatically constructed graph, near-complete paired reference truth over the same graph, and enough errors to train/evaluate KGE without substantial manual relabeling. Examples considered include AffilKG-style gold-vs-extracted graphs, CoDe-KG, KGCQual evaluation resources, and recent extraction audits.

Therefore the falsification pilot uses standard reference KGs plus **model-produced hard false triples**. This is acceptable only as a phenomenon test. A passing pilot would still require validation on a genuinely automatically constructed KG before a top-tier paper claim.

## Surviving novelty claim

The strongest defensible claim to test is:

> **Bilateral construction-noise calibration excess:** with the calibration-label noise held fixed, false observed KG edges additionally corrupt KGE representations through the training topology, causing materially worse calibration against a clean reference graph than a label-only control.

We must not claim any of the following as novel:

- KGE calibration;
- CWA versus OWA calibration gaps;
- noisy-label calibration;
- uncertain/probabilistic KGs;
- graph structural perturbations affecting calibration;
- bilateral edge noise affecting link-prediction accuracy;
- noisy triples hurting KGE performance.

## Required falsification logic

A valid experiment must compare at least:

1. **CLEAN**: clean training graph + clean calibration labels.
2. **LABEL-ONLY**: clean training graph + corrupted calibration labels.
3. **BILATERAL**: corrupted training graph + the **same corrupted calibration labels** as LABEL-ONLY.

The primary contrast is BILATERAL minus LABEL-ONLY when both are evaluated against the clean/reference labels. This isolates the additional topology channel.

If the bilateral excess is small, inconsistent, or absent on either prespecified dataset, stop. Do not rescue with new models, datasets, corruption rates, or calibration methods.

## Key references

- Tabacof, P. & Costabello, L. 2019. Probability Calibration for Knowledge Graph Embedding Models. arXiv:1912.10000.
- Safavi, T., Koutra, D. & Meij, E. 2020. Evaluating the Calibration of Knowledge Graph Embeddings for Trustworthy Link Prediction. EMNLP 2020.
- Chen, X. et al. 2021. Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning (BEUrRE).
- Zhou, Z. et al. 2023. Combating Bilateral Edge Noise for Robust Link Prediction. NeurIPS 2023.
- Liu, X., Liu, Y. & Hu, W. 2024. Knowledge Graph Error Detection with Contrastive Confidence Adaption. AAAI 2024.
- Kapoor, S. et al. 2024. Performance Evaluation of Knowledge Graph Embedding Approaches under Non-adversarial Attacks. arXiv:2407.06855.
- Yang, Y. et al. 2025. KGE Calibrator: An Efficient Probability Calibration Method of Knowledge Graph Embedding Models for Trustworthy Link Prediction. EMNLP 2025.
- Song, T. et al. 2025. Robust Knowledge Graph Embedding via Denoising. arXiv:2505.18171.
- Dang, C. et al. 2026. The Confidence Trap: Calibration Attacks for Graph Neural Networks. arXiv:2606.08467.
