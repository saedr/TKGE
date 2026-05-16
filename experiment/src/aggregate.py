import argparse
from pathlib import Path

from src.utils import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="experiment/results")
    args = ap.parse_args()
    raw = Path(args.results_root) / "raw"
    outdir = Path(args.results_root) / "aggregated"
    conds = ["Original", "Random", "Structured-low", "Structured-high"]
    metrics = {}
    orig_mrr = None
    for c in conds:
        m = read_json(raw / c / "metrics.json")
        metrics[c] = m
        if c == "Original":
            orig_mrr = m["tail_mrr"]
    summary = {"conditions": conds, "aggregation_complete": True, "results": {}}
    for c in conds:
        summary["results"][c] = {
            "tail_mrr": metrics[c]["tail_mrr"],
            "tail_hits_at_10": metrics[c]["tail_hits_at_10"],
            "delta_tail_mrr_vs_original": None if c == "Original" else metrics[c]["tail_mrr"] - orig_mrr,
            "by_coverage_bin": metrics[c]["by_coverage_bin"],
            "by_relation_frequency_bin": metrics[c]["by_relation_frequency_bin"],
        }
    write_json(outdir / "smoke_aggregate.json", summary)


if __name__ == "__main__":
    main()
