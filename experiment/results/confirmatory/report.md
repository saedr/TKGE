# Confirmatory structured-missingness replication

## Decision: **CONTINUE**

Relation-low produced uncertainty-supported overall model × mechanism interactions on both new datasets (5/6 model pairs on WN18RR and 4/6 on CoDEx-M); no supported ranking reversal occurred.

The estimand is tail-MRR degradation from the clean graph and, primarily, differential degradation versus random deletion. Positive differential degradation means that the structured mechanism harms a model more than random deletion. Model × mechanism interaction is the difference in that differential degradation between models.

## Mean tail MRR

| Dataset | Model | Mechanism | Mean MRR | 95% hierarchical CI |
|---|---|---|---|---|
| WN18RR | TransE | Original | 0.128882 | [0.120609, 0.137831] |
| WN18RR | TransE | Random | 0.130954 | [0.123648, 0.138412] |
| WN18RR | TransE | Structured-low | 0.126766 | [0.119623, 0.133879] |
| WN18RR | TransE | Relation-low | 0.114677 | [0.107734, 0.122006] |
| WN18RR | DistMult | Original | 0.220747 | [0.198559, 0.241221] |
| WN18RR | DistMult | Random | 0.135264 | [0.113727, 0.154617] |
| WN18RR | DistMult | Structured-low | 0.154828 | [0.140120, 0.170063] |
| WN18RR | DistMult | Relation-low | 0.169246 | [0.157778, 0.180383] |
| WN18RR | ComplEx | Original | 0.318190 | [0.290786, 0.342113] |
| WN18RR | ComplEx | Random | 0.217298 | [0.200210, 0.232463] |
| WN18RR | ComplEx | Structured-low | 0.223879 | [0.209524, 0.237313] |
| WN18RR | ComplEx | Relation-low | 0.240031 | [0.225991, 0.253651] |
| WN18RR | RotatE | Original | 0.002263 | [0.001588, 0.003199] |
| WN18RR | RotatE | Random | 0.003059 | [0.002123, 0.004356] |
| WN18RR | RotatE | Structured-low | 0.002989 | [0.002207, 0.004020] |
| WN18RR | RotatE | Relation-low | 0.002733 | [0.001790, 0.003837] |
| CoDEx-M | TransE | Original | 0.092906 | [0.088646, 0.097001] |
| CoDEx-M | TransE | Random | 0.094370 | [0.089920, 0.099198] |
| CoDEx-M | TransE | Structured-low | 0.094341 | [0.089421, 0.099617] |
| CoDEx-M | TransE | Relation-low | 0.102648 | [0.098604, 0.106417] |
| CoDEx-M | DistMult | Original | 0.266014 | [0.256907, 0.274934] |
| CoDEx-M | DistMult | Random | 0.262559 | [0.255626, 0.269064] |
| CoDEx-M | DistMult | Structured-low | 0.257454 | [0.250271, 0.265509] |
| CoDEx-M | DistMult | Relation-low | 0.256531 | [0.246936, 0.266093] |
| CoDEx-M | ComplEx | Original | 0.262657 | [0.254549, 0.270013] |
| CoDEx-M | ComplEx | Random | 0.258300 | [0.249585, 0.266577] |
| CoDEx-M | ComplEx | Structured-low | 0.252101 | [0.241600, 0.260380] |
| CoDEx-M | ComplEx | Relation-low | 0.248062 | [0.240836, 0.255105] |
| CoDEx-M | RotatE | Original | 0.019790 | [0.018240, 0.021506] |
| CoDEx-M | RotatE | Random | 0.019703 | [0.018297, 0.021370] |
| CoDEx-M | RotatE | Structured-low | 0.017659 | [0.016595, 0.018775] |
| CoDEx-M | RotatE | Relation-low | 0.034639 | [0.025115, 0.043433] |

## Differential degradation: structured minus random

| Dataset | Model | Structured mechanism | Differential degradation | 95% hierarchical CI |
|---|---|---|---|---|
| WN18RR | TransE | Structured-low | 0.004187 | [-0.000270, 0.008614] |
| WN18RR | TransE | Relation-low | 0.016277 | [0.009437, 0.023296] |
| WN18RR | DistMult | Structured-low | -0.019565 | [-0.048502, 0.007335] |
| WN18RR | DistMult | Relation-low | -0.033982 | [-0.057655, -0.015511] |
| WN18RR | ComplEx | Structured-low | -0.006581 | [-0.022437, 0.008582] |
| WN18RR | ComplEx | Relation-low | -0.022733 | [-0.040072, -0.007436] |
| WN18RR | RotatE | Structured-low | 0.000069 | [-0.001086, 0.001372] |
| WN18RR | RotatE | Relation-low | 0.000325 | [-0.001021, 0.001640] |
| CoDEx-M | TransE | Structured-low | 0.000029 | [-0.007362, 0.005565] |
| CoDEx-M | TransE | Relation-low | -0.008278 | [-0.013893, -0.001831] |
| CoDEx-M | DistMult | Structured-low | 0.005105 | [-0.001546, 0.011635] |
| CoDEx-M | DistMult | Relation-low | 0.006028 | [-0.002722, 0.013834] |
| CoDEx-M | ComplEx | Structured-low | 0.006199 | [-0.000027, 0.012341] |
| CoDEx-M | ComplEx | Relation-low | 0.010238 | [0.004506, 0.016428] |
| CoDEx-M | RotatE | Structured-low | 0.002044 | [0.000345, 0.003912] |
| CoDEx-M | RotatE | Relation-low | -0.014936 | [-0.023062, -0.006320] |

## Model × mechanism interactions

| Dataset | Models A−B | Structured mechanism | Interaction | 95% hierarchical CI | Supported |
|---|---|---|---|---|---|
| WN18RR | TransE−DistMult | Structured-low | 0.023752 | [-0.002618, 0.053384] | False |
| WN18RR | TransE−DistMult | Relation-low | 0.050259 | [0.031202, 0.075713] | True |
| WN18RR | TransE−ComplEx | Structured-low | 0.010768 | [-0.005037, 0.028138] | False |
| WN18RR | TransE−ComplEx | Relation-low | 0.039010 | [0.024513, 0.056873] | True |
| WN18RR | TransE−RotatE | Structured-low | 0.004118 | [-0.000671, 0.008792] | False |
| WN18RR | TransE−RotatE | Relation-low | 0.015952 | [0.009081, 0.023283] | True |
| WN18RR | DistMult−ComplEx | Structured-low | -0.012984 | [-0.053072, 0.022317] | False |
| WN18RR | DistMult−ComplEx | Relation-low | -0.011249 | [-0.044463, 0.017728] | False |
| WN18RR | DistMult−RotatE | Structured-low | -0.019634 | [-0.048587, 0.007086] | False |
| WN18RR | DistMult−RotatE | Relation-low | -0.034307 | [-0.057277, -0.015834] | True |
| WN18RR | ComplEx−RotatE | Structured-low | -0.006650 | [-0.022218, 0.008607] | False |
| WN18RR | ComplEx−RotatE | Relation-low | -0.023058 | [-0.039962, -0.007873] | True |
| CoDEx-M | TransE−DistMult | Structured-low | -0.005076 | [-0.012951, 0.005177] | False |
| CoDEx-M | TransE−DistMult | Relation-low | -0.014305 | [-0.024157, -0.005331] | True |
| CoDEx-M | TransE−ComplEx | Structured-low | -0.006169 | [-0.014107, 0.001230] | False |
| CoDEx-M | TransE−ComplEx | Relation-low | -0.018516 | [-0.028566, -0.008703] | True |
| CoDEx-M | TransE−RotatE | Structured-low | -0.002015 | [-0.010253, 0.004469] | False |
| CoDEx-M | TransE−RotatE | Relation-low | 0.006658 | [-0.001572, 0.014836] | False |
| CoDEx-M | DistMult−ComplEx | Structured-low | -0.001094 | [-0.011620, 0.007946] | False |
| CoDEx-M | DistMult−ComplEx | Relation-low | -0.004211 | [-0.015751, 0.004737] | False |
| CoDEx-M | DistMult−RotatE | Structured-low | 0.003061 | [-0.003772, 0.009638] | False |
| CoDEx-M | DistMult−RotatE | Relation-low | 0.020964 | [0.011952, 0.028829] | True |
| CoDEx-M | ComplEx−RotatE | Structured-low | 0.004155 | [-0.002599, 0.011040] | False |
| CoDEx-M | ComplEx−RotatE | Relation-low | 0.025174 | [0.014127, 0.035508] | True |

Full overall and slice-specific estimates are in `tables/interactions.csv`.

## Slice-specific results

| Dataset | Overall | Coverage low/mid/high | Relation low/mid/high |
|---|---|---|---|
| WN18RR | 3134 | 892/1167/1075 | 809/1074/1251 |
| CoDEx-M | 10311 | 2937/4748/2626 | 2790/3886/3635 |

| Dataset | Slice | Models A−B | Mechanism | Interaction | 95% CI |
|---|---|---|---|---|---|
| WN18RR | coverage_low | TransE−DistMult | Relation-low | 0.020324 | [0.003109, 0.035500] |
| WN18RR | coverage_low | TransE−ComplEx | Relation-low | 0.019522 | [0.007731, 0.031733] |
| WN18RR | coverage_low | TransE−RotatE | Relation-low | 0.027825 | [0.016434, 0.039091] |
| WN18RR | relation_low | TransE−DistMult | Relation-low | 0.029549 | [0.011152, 0.049503] |
| WN18RR | relation_low | TransE−RotatE | Relation-low | 0.080691 | [0.065356, 0.096475] |
| WN18RR | relation_low | DistMult−ComplEx | Relation-low | -0.026877 | [-0.044582, -0.009878] |
| WN18RR | relation_low | DistMult−RotatE | Relation-low | 0.051142 | [0.037823, 0.065419] |
| WN18RR | relation_low | ComplEx−RotatE | Relation-low | 0.078020 | [0.061462, 0.095154] |
| WN18RR | coverage_mid | TransE−DistMult | Relation-low | 0.057992 | [0.035119, 0.087599] |
| WN18RR | coverage_mid | TransE−ComplEx | Relation-low | 0.038074 | [0.016494, 0.066611] |
| WN18RR | coverage_mid | TransE−RotatE | Relation-low | 0.017857 | [0.003387, 0.032328] |
| WN18RR | coverage_mid | DistMult−RotatE | Relation-low | -0.040136 | [-0.073538, -0.012608] |
| WN18RR | coverage_mid | ComplEx−RotatE | Relation-low | -0.020218 | [-0.041859, -0.001691] |
| WN18RR | relation_mid | TransE−DistMult | Relation-low | 0.118892 | [0.076670, 0.174034] |
| WN18RR | relation_mid | TransE−ComplEx | Relation-low | 0.105892 | [0.070649, 0.152483] |
| WN18RR | relation_mid | TransE−RotatE | Relation-low | -0.020139 | [-0.029452, -0.010428] |
| WN18RR | relation_mid | DistMult−RotatE | Relation-low | -0.139031 | [-0.193077, -0.097552] |
| WN18RR | relation_mid | ComplEx−RotatE | Relation-low | -0.126031 | [-0.167359, -0.091739] |
| WN18RR | coverage_high | TransE−DistMult | Structured-low | 0.034813 | [0.001203, 0.074487] |
| WN18RR | coverage_high | TransE−DistMult | Relation-low | 0.066703 | [0.037137, 0.105345] |
| WN18RR | coverage_high | TransE−ComplEx | Relation-low | 0.056196 | [0.030707, 0.082145] |
| WN18RR | coverage_high | DistMult−RotatE | Structured-low | -0.035497 | [-0.074668, -0.001126] |
| WN18RR | coverage_high | DistMult−RotatE | Relation-low | -0.062671 | [-0.093910, -0.039335] |
| WN18RR | coverage_high | ComplEx−RotatE | Relation-low | -0.052164 | [-0.079372, -0.025639] |
| CoDEx-M | coverage_low | TransE−ComplEx | Relation-low | -0.013294 | [-0.025367, -0.000433] |
| CoDEx-M | coverage_low | DistMult−RotatE | Relation-low | 0.017202 | [0.004843, 0.027945] |
| CoDEx-M | coverage_low | ComplEx−RotatE | Relation-low | 0.018963 | [0.007750, 0.030303] |
| CoDEx-M | relation_low | TransE−DistMult | Relation-low | -0.040246 | [-0.062783, -0.021498] |
| CoDEx-M | relation_low | TransE−ComplEx | Relation-low | -0.049834 | [-0.070120, -0.028792] |
| CoDEx-M | relation_low | DistMult−ComplEx | Structured-low | -0.013183 | [-0.025699, -0.000196] |
| CoDEx-M | relation_low | DistMult−RotatE | Relation-low | 0.037134 | [0.024354, 0.051867] |
| CoDEx-M | relation_low | ComplEx−RotatE | Relation-low | 0.046722 | [0.030914, 0.062156] |
| CoDEx-M | coverage_mid | TransE−DistMult | Relation-low | -0.012505 | [-0.025304, -0.000842] |
| CoDEx-M | coverage_mid | TransE−ComplEx | Relation-low | -0.017473 | [-0.029561, -0.006739] |
| CoDEx-M | coverage_mid | DistMult−RotatE | Relation-low | 0.019482 | [0.008472, 0.029864] |
| CoDEx-M | coverage_mid | ComplEx−RotatE | Relation-low | 0.024450 | [0.011164, 0.037422] |
| CoDEx-M | relation_mid | TransE−DistMult | Relation-low | -0.023358 | [-0.039592, -0.008438] |
| CoDEx-M | relation_mid | TransE−ComplEx | Relation-low | -0.022295 | [-0.034352, -0.009919] |
| CoDEx-M | relation_mid | DistMult−RotatE | Relation-low | 0.024018 | [0.004666, 0.042057] |
| CoDEx-M | relation_mid | ComplEx−RotatE | Relation-low | 0.022955 | [0.012080, 0.033334] |
| CoDEx-M | coverage_high | TransE−DistMult | Relation-low | -0.020662 | [-0.033527, -0.007566] |
| CoDEx-M | coverage_high | TransE−ComplEx | Relation-low | -0.026243 | [-0.042111, -0.010221] |
| CoDEx-M | coverage_high | DistMult−RotatE | Relation-low | 0.027850 | [0.014543, 0.040442] |
| CoDEx-M | coverage_high | ComplEx−RotatE | Relation-low | 0.033431 | [0.019347, 0.046731] |
| CoDEx-M | relation_high | TransE−RotatE | Relation-low | 0.020570 | [0.010215, 0.030479] |

Complete slice estimates are in `tables/mean_mrr.csv` and `tables/differential_degradation.csv`.

## Supported model-ranking reversals

No uncertainty-supported model-ranking reversal occurred.

## FB15k-237 replication assessment

The FB15k-237 pilot showed model-dependent point differences between random and entity-coverage structured deletion: differential degradation was -0.012700 for TransE and 0.000798 for DistMult (point interaction -0.013498). That pilot did not provide the hierarchical uncertainty used here. The specific entity-coverage Structured-low pattern did not reproduce in overall MRR on either new dataset; only 3 slice-specific pairwise interaction interval(s) excluded zero. The broader model × structured-missingness phenomenon did reproduce for the prespecified relation-frequency mechanism in overall MRR on both new datasets.

The sign of the TransE-versus-bilinear interaction differs between WN18RR and CoDEx-M, so the evidence supports model dependence, not a universal claim that one family is always more fragile.

## Training-sanity limitation

WN18RR RotatE's final epoch loss exceeded its first epoch loss in 30/30 runs, and its clean MRR was 0.002263. RotatE-involving WN18RR contrasts should therefore not be treated as evidence about a competitive RotatE model. The continuation result does not depend on them: these uncertainty-supported overall Relation-low interactions remain after excluding RotatE:

| Dataset | Models A−B | Interaction | 95% CI |
|---|---|---|---|
| WN18RR | TransE−DistMult | 0.050259 | [0.031202, 0.075713] |
| WN18RR | TransE−ComplEx | 0.039010 | [0.024513, 0.056873] |
| CoDEx-M | TransE−DistMult | -0.014305 | [-0.024157, -0.005331] |
| CoDEx-M | TransE−ComplEx | -0.018516 | [-0.028566, -0.008703] |

## Frozen design and uncertainty

- Datasets: WN18RR and CoDEx-M; models: TransE, DistMult, ComplEx, RotatE.
- The three perturbed mechanisms remove 30% of training triples. Entity Structured-low uses `1/(min(deg(h),deg(t))+5)`; Relation-low uses `1/(freq(r)+5)`; both statistics come from the unperturbed training graph.
- The deletion guard leaves every training-visible entity incident to at least one retained training triple. Validation and test triples are unchanged.
- Clean models use three training seeds. Each perturbed mechanism uses three deletion realizations crossed with three training seeds.
- All original test triples are evaluated in the tail direction. The filter is the original train+validation+test union, including deleted training triples. Ties use optimistic rank: one plus the number of unfiltered candidates with strictly greater score.
- Slice assignments use quartiles of original-graph entity coverage and relation frequency and remain fixed for all conditions.
- The 95% percentile intervals use paired hierarchical bootstrap replicates that jointly resample deletion realization, training seed, and original test query within each slice. Resampled indices are shared across models and mechanisms to retain the crossed pairing.
- A reversal is supported only when both compared missingness-mechanism model-gap intervals exclude zero in opposite directions.
