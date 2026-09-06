#!/usr/bin/env python3
"""Independently validate a completed frozen confirmatory result grid."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiment.scripts.aggregate_confirmatory import (
    CONDITIONS,
    DATASETS,
    DELETION_SEEDS,
    EXPECTED_RUNS,
    MODELS,
    TRAIN_SEEDS,
    load_chunks,
)


def main(args):
    chunks = load_chunks(args.indir)
    checks = {
        "chunks": 0,
        "runs": 0,
        "perturbed_runs": 0,
        "zero_degree_violations": 0,
        "metric_recomputation_failures": 0,
        "slice_assignment_failures": 0,
        "nonfinite_rr_values": 0,
        "loss_non_decrease_runs": 0,
    }
    reference_slices = {}
    point_means = {}
    for dataset in DATASETS:
        for model in MODELS:
            chunk = chunks[(dataset, model)]
            mapping_path = Path(args.indir) / f"{dataset}_{model}.mapping.json"
            with open(mapping_path, "r", encoding="utf-8") as f:
                train_count = int(json.load(f)["train_count"])
            checks["chunks"] += 1
            assert len(chunk["runs"]) == EXPECTED_RUNS
            expected_keys = {
                (condition, deletion_seed, train_seed)
                for condition in CONDITIONS
                for deletion_seed in ([0] if condition == "Original" else DELETION_SEEDS)
                for train_seed in TRAIN_SEEDS
            }
            observed_keys = {
                (run["condition"], int(run["deletion_seed"]), int(run["train_seed"]))
                for run in chunk["runs"]
            }
            assert observed_keys == expected_keys
            for run in chunk["runs"]:
                checks["runs"] += 1
                metrics = run["metrics"]
                rr = np.asarray(metrics["rr"], dtype=float)
                assert len(rr) == chunk["num_test_queries"]
                checks["nonfinite_rr_values"] += int((~np.isfinite(rr)).sum())
                assert np.all((rr > 0.0) & (rr <= 1.0))
                if not run["loss_end"] < run["loss_start"]:
                    checks["loss_non_decrease_runs"] += 1
                if run["condition"] != "Original":
                    checks["perturbed_runs"] += 1
                    assert run["perturbation"]["removed"] == int(0.30 * train_count)
                    checks["zero_degree_violations"] += int(
                        run["perturbation"]["zero_degree_entities_after"] != 0
                    )
                coverage = np.asarray(metrics["coverage_bins"])
                relation = np.asarray(metrics["relation_bins"])
                current_slices = (tuple(coverage), tuple(relation))
                if dataset not in reference_slices:
                    reference_slices[dataset] = current_slices
                elif current_slices != reference_slices[dataset]:
                    checks["slice_assignment_failures"] += 1
                recomputed = {
                    "overall_mrr": rr.mean(),
                    **{
                        f"{value}_coverage_mrr": rr[coverage == value].mean()
                        for value in ("low", "mid", "high")
                    },
                    **{
                        f"{value}_relation_mrr": rr[relation == value].mean()
                        for value in ("low", "mid", "high")
                    },
                }
                for key, value in recomputed.items():
                    if not math.isclose(float(metrics[key]), float(value), rel_tol=0, abs_tol=1e-12):
                        checks["metric_recomputation_failures"] += 1
            for condition in CONDITIONS:
                values = [
                    run["metrics"]["overall_mrr"]
                    for run in chunk["runs"]
                    if run["condition"] == condition
                ]
                point_means[(dataset, model, condition)] = float(np.mean(values))

    with open(args.means, "r", encoding="utf-8") as f:
        mean_rows = list(csv.DictReader(f))
    overall_rows = [row for row in mean_rows if row["slice"] == "overall"]
    assert len(overall_rows) == len(DATASETS) * len(MODELS) * len(CONDITIONS)
    for row in overall_rows:
        expected = point_means[(row["dataset"], row["model"], row["condition"])]
        assert math.isclose(float(row["mean"]), expected, rel_tol=0, abs_tol=1e-12)

    assert checks["chunks"] == 8
    assert checks["runs"] == 240
    assert checks["perturbed_runs"] == 216
    for key in (
        "zero_degree_violations",
        "metric_recomputation_failures",
        "slice_assignment_failures",
        "nonfinite_rr_values",
    ):
        assert checks[key] == 0, (key, checks[key])
    checks["status"] = "PASS"
    output = json.dumps(checks, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True)
    parser.add_argument("--means", required=True)
    parser.add_argument("--out")
    main(parser.parse_args())
