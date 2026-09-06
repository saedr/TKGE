# GraphGuard execution note

The frozen TempBench phenomenon gate is executed in five deterministic inference shards for runtime only. All 1,000 selected items (or all eligible items if fewer), evidence conditions, matched controls, prompts, model settings, decoding, and statistical decision rules are identical to the frozen specification. Controls are constructed on the full selected cohort before sharding, and the final gate is applied only after exact shard coverage is reassembled.
