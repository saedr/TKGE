# DistMult 10% Chunk Report

## Status
- Completion status: complete
- Validation result: passed
- Model: DistMult
- Budget: 10%
- Evaluation kind: tail_only
- Shared filtering: original_train_valid_test

## Completion Table
- seed=11 condition=Original complete=yes missing=[]
- seed=11 condition=Random complete=yes missing=[]
- seed=11 condition=Structured-low complete=yes missing=[]
- seed=11 condition=Structured-high complete=yes missing=[]
- seed=22 condition=Original complete=yes missing=[]
- seed=22 condition=Random complete=yes missing=[]
- seed=22 condition=Structured-low complete=yes missing=[]
- seed=22 condition=Structured-high complete=yes missing=[]
- seed=33 condition=Original complete=yes missing=[]
- seed=33 condition=Random complete=yes missing=[]
- seed=33 condition=Structured-low complete=yes missing=[]
- seed=33 condition=Structured-high complete=yes missing=[]

## Overall tail_mrr / tail_hits_at_10
- Original: tail_mrr=0.281159, tail_hits_at_10=0.475000
- Random: tail_mrr=0.282479, tail_hits_at_10=0.485833
- Structured-low: tail_mrr=0.272527, tail_hits_at_10=0.466667
- Structured-high: tail_mrr=0.285621, tail_hits_at_10=0.478333

## Coverage-bin delta_tail_mrr_vs_original (mean over seeds)
- Original / low: 0.000000
- Original / mid: 0.000000
- Original / high: 0.000000
- Random / low: -0.018508
- Random / mid: 0.006861
- Random / high: 0.010066
- Structured-low / low: -0.030214
- Structured-low / mid: 0.003004
- Structured-low / high: -0.010324
- Structured-high / low: -0.011116
- Structured-high / mid: 0.009374
- Structured-high / high: 0.010213

## Relation-frequency-bin delta_tail_mrr_vs_original (mean over seeds)
- Original / low-frequency: 0.000000
- Original / mid-frequency: 0.000000
- Original / high-frequency: 0.000000
- Random / low-frequency: -0.006985
- Random / mid-frequency: -0.006214
- Random / high-frequency: 0.024700
- Structured-low / low-frequency: -0.003498
- Structured-low / mid-frequency: -0.003404
- Structured-low / high-frequency: -0.024221
- Structured-high / low-frequency: -0.007958
- Structured-high / mid-frequency: 0.015721
- Structured-high / high-frequency: -0.005402

## Jaccard@10 by coverage/relation-frequency bin
- Original: coverage={'low': 0.1672101662813737, 'mid': 0.19120282996443988, 'high': 0.22285584184190996}, relation_frequency={'low-frequency': 0.20790542123650665, 'mid-frequency': 0.18371534667021386, 'high-frequency': 0.19689365272956602}
- Random: coverage={'low': 0.16209888551993815, 'mid': 0.20855889122916987, 'high': 0.2372015062107941}, relation_frequency={'low-frequency': 0.20893787458787613, 'mid-frequency': 0.19683167803024051, 'high-frequency': 0.21369588157513855}
- Structured-low: coverage={'low': 0.1577614155710131, 'mid': 0.20281415508582692, 'high': 0.23799995469500113}, relation_frequency={'low-frequency': 0.21779607762656528, 'mid-frequency': 0.18821592077577887, 'high-frequency': 0.20686595969103708}
- Structured-high: coverage={'low': 0.1619542474496035, 'mid': 0.1885810102063972, 'high': 0.22786387760845964}, relation_frequency={'low-frequency': 0.19850617332455298, 'mid-frequency': 0.19109171997063465, 'high-frequency': 0.18621638767149604}