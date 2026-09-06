#!/usr/bin/env python3
"""Focused regression tests for the frozen confirmatory implementation."""

from collections import Counter

import numpy as np
import torch

from experiment.scripts.aggregate_confirmatory import bootstrap_condition_means
from experiment.scripts.run_confirmatory_chunk import (
    CONDITIONS,
    define_slices,
    evaluate_rr,
    make_model,
    perturb_relation_low,
)
from experiment.src.perturb import perturb_training


def test_batched_scores():
    torch.manual_seed(7)
    h = torch.tensor([0, 2])
    r = torch.tensor([0, 1])
    candidates = torch.arange(5)
    for name in ("TransE", "DistMult", "ComplEx", "RotatE"):
        model = make_model(name, 5, 2, 4)
        actual = model.score_tail_batch(h, r, candidates)
        expected = torch.stack([
            model.score(
                torch.full((len(candidates),), int(hh), dtype=torch.long),
                torch.full((len(candidates),), int(rr), dtype=torch.long),
                candidates,
            )
            for hh, rr in zip(h, r)
        ])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


class FixedScores(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scores", torch.tensor([0.1, 0.8, 0.7, 0.9]))

    def score(self, h, r, t):
        return self.scores[t]

    def score_tail_batch(self, h, r, candidates):
        return self.scores[candidates].expand(len(h), -1)


def test_filtered_rank_and_fixed_slices():
    test = [(0, 0, 2), (1, 0, 1)]
    original_filter = test + [(0, 0, 3)]
    degree = Counter({0: 5, 1: 3, 2: 2, 3: 1})
    relfreq = Counter({0: 10})
    coverage, relation, metadata = define_slices(test, degree, relfreq)
    metrics = evaluate_rr(
        FixedScores(), test, original_filter, 4, coverage, relation, torch.device("cpu"),
        query_batch_size=2, candidate_chunk_size=2,
    )
    # Query 0: candidate 3 is filtered, candidate 1 remains above target => rank 2.
    # Query 1: candidate 3 remains above target => rank 2.
    np.testing.assert_allclose(metrics["rr"], [0.5, 0.5])
    assert metadata["source"] == "original_training_graph_before_perturbation"


def test_deletion_guards():
    train = [(0, 0, 1), (1, 0, 2), (2, 1, 3), (3, 1, 0), (0, 1, 2), (1, 1, 3)]
    degree = Counter()
    relfreq = Counter()
    for h, r, t in train:
        degree[h] += 1
        degree[t] += 1
        relfreq[r] += 1
    for condition in ("Random", "Structured-low"):
        kept, meta = perturb_training(train, condition, 0.30, degree, 5, 101)
        assert len(kept) == len(train) - int(0.30 * len(train))
        assert meta["zero_degree_entities_after"] == 0
    kept, meta = perturb_relation_low(train, 0.30, degree, relfreq, 5, 101)
    assert len(kept) == len(train) - int(0.30 * len(train))
    assert meta["zero_degree_entities_after"] == 0


def test_hierarchical_bootstrap_pairing():
    rng = np.random.default_rng(1)
    arrays = {}
    base = np.arange(18, dtype=float).reshape(3, 3, 2) / 100.0
    for model in ("TransE", "DistMult", "ComplEx", "RotatE"):
        arrays[model] = {
            "Original": base[:1].copy(),
            "Random": base.copy(),
            "Structured-low": base.copy(),
            "Relation-low": base.copy(),
        }
    draws = bootstrap_condition_means(arrays, np.array([0, 1]), 20, rng)
    assert draws.shape == (20, 4, len(CONDITIONS))
    # Shared resampling makes identical inputs exactly identical across models/mechanisms.
    np.testing.assert_allclose(draws[:, 0, 1:], draws[:, 3, 1:])
    np.testing.assert_allclose(draws[:, :, 1], draws[:, :, 2])


def main():
    test_batched_scores()
    test_filtered_rank_and_fixed_slices()
    test_deletion_guards()
    test_hierarchical_bootstrap_pairing()
    print("confirmatory regression tests: PASS")


if __name__ == "__main__":
    main()
