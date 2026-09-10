# PASSLEAF original-code crossed 3x3 replay

This package repeats the earlier NL27K LOWER/NOMINAL/UPPER × seeds {11,23,37} diagnostic using the original unKR PASSLEAF classes and the original PASSLEAF checkpoint-selection rule (`Eval_MAE`, lower is better).

The previous package selected checkpoints by `Eval_wmr`; that is the implementation error this replay corrects. World construction, query construction, raw top-1 evaluation, seeds, and analysis estimands are otherwise preserved.

The replay uses the local unKR checkout that already produced the successful baseline run. It looks for `unKR` either inside the TKGE repository or next to it.

Run from the TKGE repository root:

    bash experiment/identified_ranking_passleaf_original_replay/run_3x3.sh "$PWD" cpu

The runner is resumable at the completed world×seed evaluation level. Results are written under `experiment/identified_ranking_passleaf_original_replay/runs/` and the final aggregate to `RESULT_3x3.json`.

The decisive gate is frozen in `REPLAY_SPEC.md`: usable NOMINAL seed-11 performance plus `DeltaDisagree >= 0.10`. Variance-share diagnostics are reported but cannot rescue a failed primary gate.
