# Frozen GraphGuard TempBench phenomenon gate

- Dataset: TempBench revision `ad8ea76`, test split only.
- Exclude interval questions.
- Require both typed negatives to be functional, single-edge replacements, and to target the same gold edge.
- Evaluate at most 1,000 eligible questions selected deterministically before inference.
- Model: `Qwen/Qwen3-1.7B`, greedy decoding, no fine-tuning.
- Conditions: gold, provided relation-matched distractor, provided stale fact, matched random-object control, matched random-time control.
- Primary population: questions answered correctly under gold evidence.
- Continue only if at least one plausible intervention flips >=10% of gold-correct decisions and its flip rate exceeds the matched random control with a paired-bootstrap 95% CI entirely above zero.
- Minimum 100 gold-correct questions required for interpretation.
- If the gate fails, do not build GraphGuard from this phenomenon without a new independent rationale or dataset.
