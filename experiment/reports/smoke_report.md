# DistMult Smoke Experiment Report

## Status
- Smoke completed end-to-end: yes
- Validation passed: yes
- Branch: work
- Data layout used: flat data with runtime symlinks to data/FB15k-237
- Evaluation kind: tail_only
- Shared filtering: original_train_valid_test

## Design
- Dataset: FB15k-237
- Model: DistMult
- Seed: 11
- Budget: 30%
- Conditions: Original, Random, Structured-low, Structured-high
- Note: this is smoke only, not the 3-seed pilot.

## Perturbation Checks
| condition | removed_triples | zero_degree_entities_after | skipped_deletions | would_isolate_count |
|---|---:|---:|---:|---:|
| Original | 0 | 0 | 0 | 0 |
| Random | 81634 | 0 | 0 | 135 |
| Structured-low | 81634 | 0 | 0 | 949 |
| Structured-high | 81634 | 0 | 0 | 21 |

## Runtime
| condition | data_loading_sec | perturbation_sec | training_sec | evaluation_sec | total_sec |
|---|---:|---:|---:|---:|---:|
| Original | - | - | - | - | - |
| Random | - | - | - | - | - |
| Structured-low | - | - | - | - | - |
| Structured-high | - | - | - | - | - |

## Training Sanity
| condition | initial_loss | final_loss | roughly_decreasing |
|---|---:|---:|---|
| Original | - | - | True |
| Random | - | - | True |
| Structured-low | - | - | True |
| Structured-high | - | - | True |

## Overall Tail-Only Metrics
| condition | tail_mrr | tail_hits_at_10 | delta_tail_mrr_vs_original |
|---|---|---|---|
| Original | 0.287781 | 0.477500 | - |
| Random | 0.271210 | 0.465000 | -0.016571 |
| Structured-low | 0.265461 | 0.437500 | -0.022320 |
| Structured-high | 0.278540 | 0.467500 | -0.009241 |

## Coverage-Bin Metrics
| condition | bin | tail_mrr | tail_hits_at_10 | delta_tail_mrr_vs_original |
|---|---|---|---|---|
| Original | low | 0.280905 | 0.390000 | - |
| Original | mid | 0.268672 | 0.450000 | - |
| Original | high | 0.332874 | 0.620000 | - |
| Random | low | 0.243399 | 0.350000 | -0.037506 |
| Random | mid | 0.267191 | 0.465000 | -0.001481 |
| Random | high | 0.307059 | 0.580000 | -0.025815 |
| Structured-low | low | 0.230751 | 0.290000 | -0.050154 |
| Structured-low | mid | 0.259296 | 0.435000 | -0.009376 |
| Structured-low | high | 0.312499 | 0.590000 | -0.020375 |
| Structured-high | low | 0.271828 | 0.380000 | -0.009078 |
| Structured-high | mid | 0.260094 | 0.475000 | -0.008578 |
| Structured-high | high | 0.322144 | 0.540000 | -0.010730 |

## Relation-Frequency-Bin Metrics
| condition | bin | tail_mrr | tail_hits_at_10 | delta_tail_mrr_vs_original |
|---|---|---|---|---|
| Original | low-frequency | 0.365588 | 0.564356 | - |
| Original | mid-frequency | 0.300980 | 0.462312 | - |
| Original | high-frequency | 0.182929 | 0.420000 | - |
| Random | low-frequency | 0.360141 | 0.544554 | -0.005447 |
| Random | mid-frequency | 0.290990 | 0.452261 | -0.009991 |
| Random | high-frequency | 0.142029 | 0.410000 | -0.040900 |
| Structured-low | low-frequency | 0.303916 | 0.445545 | -0.061672 |
| Structured-low | mid-frequency | 0.289717 | 0.452261 | -0.011264 |
| Structured-low | high-frequency | 0.178351 | 0.400000 | -0.004579 |
| Structured-high | low-frequency | 0.366876 | 0.564356 | 0.001288 |
| Structured-high | mid-frequency | 0.280334 | 0.462312 | -0.020646 |
| Structured-high | high-frequency | 0.185750 | 0.380000 | 0.002821 |

## Interpretation
- Did structured missingness produce different reliability profiles in smoke? Yes, structured settings changed reliability by bin versus Random.
- Did Structured-low disproportionately affect low-coverage triples relative to Random? Yes in this run, low-coverage bins dropped more under Structured-low.
- Are aggregate metrics sufficient, or do bin-level metrics reveal differences? Bin-level metrics reveal differences hidden by aggregates.
- Be cautious: one seed, smoke subset, tail-only evaluation.

## Next Step
Run the 3-seed DistMult 30% pilot before adding more models.

## Artifact Locations
- Raw artifacts: experiment/results/raw
- Aggregated artifacts: experiment/results/aggregated
- Figures: experiment/results/figures
