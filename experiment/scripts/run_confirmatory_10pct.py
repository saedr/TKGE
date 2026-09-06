#!/usr/bin/env python3
"""Run one dataset/model chunk for the frozen 10% replication."""
import argparse, json, time
from pathlib import Path
import torch

from experiment.src.data import load_dataset
from experiment.src.perturb import perturb_training
from experiment.src.train import train_kge
from experiment.src.utils import set_seed
from experiment.scripts.run_confirmatory_chunk import (
    TRAIN_SEEDS, DELETION_SEEDS, CONDITIONS, EXPECTED_RUNS,
    make_model, perturb_relation_low, define_slices, evaluate_rr,
    atomic_json, progress,
)

BUDGET = 0.10
MODELS = ["TransE", "DistMult", "ComplEx"]


def run(args):
    if args.model not in MODELS:
        raise ValueError(args.model)
    cfg = {
        "seed": 11,
        "dataset": {"root": f"data/{args.dataset}", "train": "train.txt", "valid": "valid.txt", "test": "test.txt", "allow_fallback": False},
        "model": {"name": args.model, "embedding_dim": 64},
        "training": {"epochs": 8, "batch_size": 512, "lr": 0.01, "negatives_per_positive": 5, "checkpoint_name": "checkpoint.pt"},
        "evaluation": {"test_queries": "all", "tail_only": True, "tie_handling": "optimistic: 1 + count(scores > target_score)"},
        "perturbation": {"budget": BUDGET, "smoothing_c": 5},
        "runtime": {"device": "cpu"},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.progress_log) if args.progress_log else out_path.with_suffix(".progress.log")
    ds = load_dataset(cfg, str(out_path.with_suffix(".mapping.json")))
    filter_triples = ds["train"] + ds["valid"] + ds["test"]
    coverage_bins, relation_bins, slice_metadata = define_slices(ds["test"], ds["orig_degree"], ds["orig_relfreq"])
    device = torch.device("cpu")
    expected = {
        "dataset": args.dataset, "model": args.model, "budget": BUDGET,
        "embedding_dim": 64, "epochs": 8, "num_test_queries": len(ds["test"]),
        "train_seeds": TRAIN_SEEDS, "deletion_seeds": DELETION_SEEDS, "conditions": CONDITIONS,
    }
    if out_path.exists():
        result = json.loads(out_path.read_text(encoding="utf-8"))
        for k, v in expected.items():
            if result.get(k) != v:
                raise RuntimeError(f"Refusing incompatible resume: {k}={result.get(k)!r}, expected {v!r}")
    else:
        result = {
            **expected,
            "filter_membership": "all original train+valid+test triples; deleted training triples remain filters",
            "tie_handling": "optimistic: rank = 1 + number of unfiltered candidates with strictly greater score",
            "slice_metadata": slice_metadata, "runs": [], "complete": False,
        }
        atomic_json(out_path, result)
    completed = {(x["condition"], int(x["deletion_seed"]), int(x["train_seed"])) for x in result["runs"]}
    progress(log_path, f"START 10pct dataset={args.dataset} model={args.model} completed={len(completed)}/{EXPECTED_RUNS}")

    for condition in CONDITIONS:
        deletion_seeds = [0] if condition == "Original" else DELETION_SEEDS
        for deletion_seed in deletion_seeds:
            if condition == "Original":
                train_cond = ds["train"]
                perturb_meta = {"condition": "Original", "removed": 0, "zero_degree_entities_after": 0}
            elif condition in ("Random", "Structured-low"):
                train_cond, perturb_meta = perturb_training(
                    ds["train"], condition, BUDGET, ds["orig_degree"], 5, deletion_seed,
                    valid_triples=ds["valid"], test_triples=ds["test"],
                )
            elif condition == "Relation-low":
                train_cond, perturb_meta = perturb_relation_low(
                    ds["train"], BUDGET, ds["orig_degree"], ds["orig_relfreq"], 5, deletion_seed,
                )
            else:
                raise ValueError(condition)
            for train_seed in TRAIN_SEEDS:
                key = (condition, deletion_seed, train_seed)
                if key in completed:
                    progress(log_path, f"SKIP completed condition={condition} deletion_seed={deletion_seed} train_seed={train_seed}")
                    continue
                progress(log_path, f"RUN condition={condition} deletion_seed={deletion_seed} train_seed={train_seed}")
                start = time.time()
                set_seed(train_seed)
                model = make_model(args.model, ds["num_entities"], ds["num_relations"], 64)
                losses = train_kge(model, train_cond, ds["num_entities"], cfg, device)
                metrics = evaluate_rr(model, ds["test"], filter_triples, ds["num_entities"], coverage_bins, relation_bins, device)
                result["runs"].append({
                    "condition": condition, "deletion_seed": deletion_seed, "train_seed": train_seed,
                    "metrics": metrics, "perturbation": perturb_meta,
                    "loss_start": float(losses[0]), "loss_end": float(losses[-1]), "runtime_sec": time.time() - start,
                })
                completed.add(key)
                result["complete"] = len(completed) == EXPECTED_RUNS
                atomic_json(out_path, result)
                progress(log_path, f"DONE condition={condition} deletion_seed={deletion_seed} train_seed={train_seed} overall_mrr={metrics['overall_mrr']:.8f} completed={len(completed)}/{EXPECTED_RUNS}")
    if len(completed) != EXPECTED_RUNS:
        raise RuntimeError(f"Incomplete chunk: {len(completed)}/{EXPECTED_RUNS}")
    result["complete"] = True
    atomic_json(out_path, result)
    progress(log_path, f"COMPLETE 10pct dataset={args.dataset} model={args.model} runs={len(completed)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["WN18RR", "CoDEx-M"], required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress-log")
    run(ap.parse_args())
