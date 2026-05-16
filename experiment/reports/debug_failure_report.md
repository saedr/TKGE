# Debug Failure Report

## Gate/Chunk
- ComplEx chunks (10% and 30%)

## Command
- `python experiment/scripts/resume_complex_chunks.py`

## Failure type
- Runtime/throughput blocker (no code traceback)

## Completion table at stop
- budget | seed | condition | complete | missing files
- 10 | 11 | Original | yes | []
- 10 | 11 | Random | yes | []
- 10 | 11 | Structured-low | yes | []
- 10 | 11 | Structured-high | yes | []
- 10 | 22 | Original | yes | []
- 10 | 22 | Random | yes | []
- 10 | 22 | Structured-low | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 10 | 22 | Structured-high | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 10 | 33 | Original | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 10 | 33 | Random | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 10 | 33 | Structured-low | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 10 | 33 | Structured-high | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 11 | Original | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 11 | Random | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 11 | Structured-low | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 11 | Structured-high | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 22 | Original | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 22 | Random | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 22 | Structured-low | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 22 | Structured-high | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 33 | Original | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 33 | Random | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 33 | Structured-low | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']
- 30 | 33 | Structured-high | no | ['metrics.json', 'perturbation.json', 'runtime.json', 'training.json', 'top10_tail_detailed.json']

## Summary
- Completed runs: 6/24
- Missing runs: 18/24
- Do not claim chunk-level conclusions from this partial state.

## Design/validation integrity
- No weakening of validation or experiment invariants.
- Tail-only evaluation, shared filtering, and zero-degree constraints unchanged.

## Next step
- Re-run `python experiment/scripts/resume_complex_chunks.py` in a longer execution window, then run full ComplEx 10% and 30% validation/report generation.