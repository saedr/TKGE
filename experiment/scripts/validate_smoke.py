#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


def loadj(p):
    return json.loads(Path(p).read_text())


def main():
    root = Path("experiment/results")
    conds = ["Original", "Random", "Structured-low", "Structured-high"]
    for c in conds:
        assert (root / "raw" / c / "metrics.json").exists(), f"missing metrics for {c}"

    removed = [loadj(root / "raw" / c / "perturbation.json")["removed"] for c in conds[1:]]
    assert len(set(removed)) == 1, "deletion counts differ"

    for c in conds:
        p = loadj(root / "raw" / c / "perturbation.json")
        assert "zero_degree_entities_after" in p
        assert "would_have_isolated_entities_count" in p
        assert p["zero_degree_entities_after"] == 0

        m = loadj(root / "raw" / c / "metrics.json")
        assert m["coverage_bins_source"] == "original_train_graph"
        assert m["relation_frequency_bins_source"] == "original_train_graph"
        assert m["shared_filtering"] == "original_train_valid_test"
        assert m["evaluation_kind"] == "tail_only"
        assert "tail_mrr" in m and "tail_hits_at_10" in m and "by_coverage_bin" in m

        top = np.load(root / "raw" / c / "top10_tail_ids.npy")
        assert top.ndim == 2 and top.shape[1] == 10

    agg = loadj(root / "aggregated" / "smoke_aggregate.json")
    assert agg.get("aggregation_complete") is True
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
