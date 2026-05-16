#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment.src.perturb import perturb_training


def _run(condition, triples, budget, seed=0):
    degree = {}
    for h, _, t in triples:
        degree[h] = degree.get(h, 0) + 1
        degree[t] = degree.get(t, 0) + 1
    kept, meta = perturb_training(
        train_triples=triples,
        condition=condition,
        budget=budget,
        orig_degree=degree,
        smoothing_c=1.0,
        seed=seed,
        valid_triples=[],
        test_triples=[],
    )
    return kept, meta


def main():
    # 0 has degree 3, 1 has degree 2, 2 has degree 1, 3 has degree 2
    triples = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (1, 0, 3)]
    budget = 0.25  # exactly one deletion feasible for every condition with guard

    for cond in ["Random", "Structured-low", "Structured-high"]:
        kept, meta = _run(cond, triples, budget, seed=17)
        assert len(kept) == 3
        assert meta["removed"] == 1
        assert meta["requested_removed"] == 1
        assert meta["zero_degree_entities_after"] == 0

    removed = [_run(cond, triples, budget, seed=17)[1]["removed"] for cond in ["Random", "Structured-low", "Structured-high"]]
    assert len(set(removed)) == 1

    # Impossible budget with guard: every edge touches a degree-1 leaf, so no deletion is legal.
    impossible_triples = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]
    try:
        _run("Structured-low", impossible_triples, budget=0.25, seed=5)
        raise AssertionError("expected impossible budget error")
    except RuntimeError as e:
        assert "Unable to satisfy deletion budget" in str(e)

    print("PERTURB_GUARD_TEST_PASSED")


if __name__ == "__main__":
    main()
