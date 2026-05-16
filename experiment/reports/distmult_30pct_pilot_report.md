# DistMult 30% Pilot Report

## Status
- Pilot completed end-to-end: yes
- Validation passed: yes
- Model: DistMult
- Seeds: 11, 22, 33
- Budget: 30%
- Evaluation kind: tail_only
- Shared filtering: original_train_valid_test
- Bin-specific stability now truly bin-separated: yes

## Overall Tail-Only Metrics
- Original: mean_tail_mrr=0.281159, mean_tail_hits_at_10=0.475000, delta_vs_original=0.000000
- Random: mean_tail_mrr=0.272052, mean_tail_hits_at_10=0.468333, delta_vs_original=-0.009108
- Structured-low: mean_tail_mrr=0.271254, mean_tail_hits_at_10=0.445000, delta_vs_original=-0.009906
- Structured-high: mean_tail_mrr=0.273811, mean_tail_hits_at_10=0.450000, delta_vs_original=-0.007349

## Predictive Stability (Jaccard@10)
- Original: overall=0.193118, coverage={'low': 0.1672101662813737, 'mid': 0.19120282996443988, 'high': 0.22285584184190996}, relation_freq={'low-frequency': 0.20790542123650665, 'mid-frequency': 0.18371534667021386, 'high-frequency': 0.19689365272956602}
- Random: overall=0.190463, coverage={'low': 0.15657303738573086, 'mid': 0.1898829169340005, 'high': 0.2255141166596275}, relation_freq={'low-frequency': 0.1950755520782802, 'mid-frequency': 0.19156774035268673, 'high-frequency': 0.18360687701244974}
- Structured-low: overall=0.192940, coverage={'low': 0.1469851805687719, 'mid': 0.19433003500340962, 'high': 0.2361155084452298}, relation_freq={'low-frequency': 0.18763530431678668, 'mid-frequency': 0.1947853758649222, 'high-frequency': 0.19462620368967118}
- Structured-high: overall=0.187983, coverage={'low': 0.16566864450765378, 'mid': 0.1896858068258997, 'high': 0.20689057617076193}, relation_freq={'low-frequency': 0.19118466043181734, 'mid-frequency': 0.18634080832456612, 'high-frequency': 0.18801611872819304}

## Interpretation
- DistMult-only, 30%-budget-only, tail-only evaluation; this is not a social-fairness claim.
- Bin-level metrics and stability are computed on explicit bin subsets and can differ from aggregate values.