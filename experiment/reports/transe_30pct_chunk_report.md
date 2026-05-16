# TransE 30% Chunk Report

## Status
- completion status: complete
- validation status: passed
- model: TransE
- budget: 30%
- evaluation kind: tail_only
- shared filtering: original_train_valid_test

## Overall tail_mrr and tail_hits_at_10 by condition
- Original: mean_tail_mrr=0.186833, mean_tail_hits_at_10=0.300000, delta_tail_mrr_vs_original=0.000000
- Random: mean_tail_mrr=0.164617, mean_tail_hits_at_10=0.287500, delta_tail_mrr_vs_original=-0.022216
- Structured-low: mean_tail_mrr=0.177317, mean_tail_hits_at_10=0.280000, delta_tail_mrr_vs_original=-0.009516
- Structured-high: mean_tail_mrr=0.169782, mean_tail_hits_at_10=0.285833, delta_tail_mrr_vs_original=-0.017051

## Jaccard@10 stability by coverage bin and relation-frequency bin
- Original: coverage={'low': 0.10195401147413531, 'mid': 0.09720336852689793, 'high': 0.13806992972627646}, relation_frequency={'low-frequency': 0.17752943429613263, 'mid-frequency': 0.09745302353078074, 'high-frequency': 0.061194432788860026}
- Random: coverage={'low': 0.09891335759756811, 'mid': 0.10252719202022606, 'high': 0.1373381007823732}, relation_frequency={'low-frequency': 0.17674853651005445, 'mid-frequency': 0.10281907050580137, 'high-frequency': 0.05817987023869377}
- Structured-low: coverage={'low': 0.09951699086374009, 'mid': 0.10685837067416015, 'high': 0.1461274286429085}, relation_frequency={'low-frequency': 0.17916742541240593, 'mid-frequency': 0.10573226750086279, 'high-frequency': 0.06799484886172193}
- Structured-high: coverage={'low': 0.10529986954909554, 'mid': 0.10224913373520184, 'high': 0.1359052056188279}, relation_frequency={'low-frequency': 0.18159753409423887, 'mid-frequency': 0.10477237076953927, 'high-frequency': 0.053792815371762746}

## Cautious interpretation
- Structured-low degrades low-coverage more than Random at 30%? no (Structured-low=-0.018818, Random=-0.029750).
- Structured-low degrades low-frequency more than Random at 30%? no (Structured-low=-0.009085, Random=-0.012414).
- Aggregate metrics can hide bin-level differences; bin deltas and bin-specific stability provide finer reliability profile detail.
- Is 30% stronger than 10%? low-coverage Structured-low delta: 10%=-0.013841, 30%=-0.018818; low-frequency Structured-low delta: 10%=-0.020269, 30%=-0.009085.
- Scope: TransE-only, tail-only evaluation; not a social-fairness experiment.