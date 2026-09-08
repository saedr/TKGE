# Pre-result Implementation Protocol

Frozen before implementation execution/results: 2026-09-08.

This file resolves software details that were not numerically specified in `FROZEN_SPEC.md`. It does not alter the scientific gates.

## Why a port is required

The official UnKGCP UKGE folder pins TensorFlow 1.5.0 and scikit-learn 0.19.1. Its current `run_nl27k.py` uses a 3,000-epoch TensorFlow/W&B sweep. Those packages are not reproducibly installable on a current GitHub Actions Python image. We therefore port the exact UKGE-logistic score family and loss ingredients to current PyTorch rather than silently changing the scientific question.

## Pinned sources

- UKGE data and original implementation: `stasl0217/UKGE@5049b84501657abc44c36dd7d2db5f8519e5ee8f`
- UnKGCP interval implementation: `0sidewalkenforcer0/UnKGCP@6cdf658be1825be0b369084859dff129ae3c113f`

The exact `train.tsv`, `val.tsv`, `test.tsv`, and `softlogic.tsv` files are retrieved from the pinned UKGE commit.

## UKGE point predictor port

Preserved from the official UKGE-logistic implementation:

- DistMult latent score `sum(h * r * t)`;
- learned scalar logistic transform `sigmoid(a * score + b)`;
- entity/relation embeddings initialized from a truncated Normal with mean 0 and SD 0.3;
- positive-fact squared error against the observed confidence;
- negative head and tail corruptions trained toward confidence 0;
- PSL/softlogic lower-bound penalty with coefficient 0.2;
- Adam optimizer with learning rate 0.001.

Fixed computational settings for the modern port:

- embedding dimension 128;
- batch size 4096;
- two head and two tail negative corruptions per positive;
- maximum 200 epochs;
- validation MSE checked every 5 epochs;
- retain the checkpoint with minimum validation MSE;
- stop after 8 consecutive validation checks without improvement;
- seed 0.

These settings are fixed before any interval or downstream-ranking result is inspected. They will not be tuned if the benchmark or phenomenon gate fails.

The larger batch and smaller negative count are computational substitutions for the obsolete reference training loop. The study is therefore a phenomenon falsification using the published UKGE model family, not a bit-for-bit reproduction claim. The frozen conformal coverage and ambiguity gates protect against interpreting an unusable point predictor as scientific evidence.

## Adaptive conformal intervals

For positive uncertain facts, port exactly the positive-sample branch in the pinned `UKGE-master/test_adaptive_CP.py`:

1. validation residual `|score - confidence|`;
2. uncertainty equal to Bernoulli entropy of the score, clipped away from 0;
3. for CN15k only, apply the upstream `0.5 * score + 0.5` transformation inside the entropy calculation;
4. normalized nonconformity = residual / entropy uncertainty;
5. finite-sample 90% quantile using the upstream ceiling correction and `higher` quantile;
6. test interval `[max(0, score-qhat*u), min(1, score+qhat*u)]`.

Only the positive-fact conformal branch is used because the frozen uncertainty pool consists of observed uncertain facts, not synthetic negative triples.

## Downstream ComplEx

Use the exact frozen hyperparameters in `FROZEN_SPEC.md` and the same ComplEx score implementation already present in TKGE. Each admissible graph world is trained from the same seed-11 initialization. The nominal seed-multiplicity control uses seeds 11, 23, 37, 53, and 71.

## Execution interpretation

A dependency/download/parser failure is an engineering failure and may be repaired without changing the scientific spec.

A run that reaches the frozen benchmark/phenomenon gates is scientifically interpretable. After that point, no changes to models, thresholds, datasets, interval level, witness-world construction, or metrics are allowed as a rescue.