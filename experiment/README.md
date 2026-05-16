# DistMult Smoke Experiment (FB15k-237)

Custom PyTorch-only (no PyKEEN) smoke experiment to compare reliability profiles under equal-budget missingness.

## Run

```bash
pip install -r experiment/requirements.txt
bash experiment/scripts/run_smoke.sh
```

## Notes
- Requires local files at `data/FB15k-237/train.txt`, `valid.txt`, `test.txt`.
- No auto-download is performed.
- Outputs are written to `experiment/results/` and are git-ignored (except `.gitkeep` placeholders).
- Metrics are explicitly tail-only: `tail_mrr`, `tail_hits_at_10`.
- Stability note is included; cross-seed Jaccard@10 is deferred to pilot (3+ seeds).
