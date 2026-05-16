# TransE 10% Chunk Report

## Status
- Completion status: complete
- Validation status: passed
- Model: TransE
- Budget: 10%
- Evaluation kind: tail_only
- Shared filtering: original_train_valid_test
- Note: TransE only, 10% budget only, tail-only, not a social-fairness experiment.

## Overall Tail-Only Metrics
- Original: mean_tail_mrr=0.186833, mean_tail_hits_at_10=0.300000, delta_tail_mrr_vs_original=0.000000
- Random: mean_tail_mrr=0.163640, mean_tail_hits_at_10=0.283333, delta_tail_mrr_vs_original=-0.023193
- Structured-low: mean_tail_mrr=0.171222, mean_tail_hits_at_10=0.290000, delta_tail_mrr_vs_original=-0.015611
- Structured-high: mean_tail_mrr=0.168924, mean_tail_hits_at_10=0.277500, delta_tail_mrr_vs_original=-0.017909

## Predictive Stability
- Original: coverage={'low': 0.10195401147413531, 'mid': 0.09720336852689793, 'high': 0.13806992972627646} relation_frequency={'low-frequency': 0.17752943429613263, 'mid-frequency': 0.09745302353078074, 'high-frequency': 0.061194432788860026}
- Random: coverage={'low': 0.09584848072464171, 'mid': 0.1046545889038149, 'high': 0.1378145933873488} relation_frequency={'low-frequency': 0.1782631199760219, 'mid-frequency': 0.10121802150832752, 'high-frequency': 0.06150263794226644}
- Structured-low: coverage={'low': 0.10315044117675697, 'mid': 0.10452389200067219, 'high': 0.1381590915104847} relation_frequency={'low-frequency': 0.17506098753991345, 'mid-frequency': 0.10602557800233486, 'high-frequency': 0.0625548190486271}
- Structured-high: coverage={'low': 0.0953070126754337, 'mid': 0.09959376164484524, 'high': 0.1366841168234357} relation_frequency={'low-frequency': 0.16618812790225593, 'mid-frequency': 0.10304290800813665, 'high-frequency': 0.05827325667108949}

## Cautious Interpretation
- Structured missingness can differ from random missingness under equal budget in both aggregate and bin-level behavior; inspect JSON for full tables.
- This is an illustrative reliability experiment on FB15k-237 with tail-only evaluation, not a protected-attribute fairness experiment.