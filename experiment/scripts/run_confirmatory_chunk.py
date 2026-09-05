#!/usr/bin/env python3
"""Run one resumable dataset/model chunk of the frozen confirmatory grid."""

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiment.src.data import load_dataset
from experiment.src.model import ComplEx, DistMult, TransE
from experiment.src.perturb import perturb_training
from experiment.src.train import train_kge
from experiment.src.utils import set_seed


TRAIN_SEEDS = [11, 22, 33]
DELETION_SEEDS = [101, 202, 303]
CONDITIONS = ["Original", "Random", "Structured-low", "Relation-low"]
EXPECTED_RUNS = 3 + 3 * 3 * 3


class RotatE(torch.nn.Module):
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.ent_re = torch.nn.Embedding(num_entities, emb_dim)
        self.ent_im = torch.nn.Embedding(num_entities, emb_dim)
        self.rel_phase = torch.nn.Embedding(num_relations, emb_dim)
        torch.nn.init.xavier_uniform_(self.ent_re.weight)
        torch.nn.init.xavier_uniform_(self.ent_im.weight)
        torch.nn.init.uniform_(self.rel_phase.weight, a=-np.pi, b=np.pi)

    def score(self, h, r, t):
        phase = self.rel_phase(r)
        rr = torch.cos(phase)
        ri = torch.sin(phase)
        hr = self.ent_re(h)
        hi = self.ent_im(h)
        tr = self.ent_re(t)
        ti = self.ent_im(t)
        rot_r = hr * rr - hi * ri
        rot_i = hr * ri + hi * rr
        return -torch.sqrt((rot_r - tr) ** 2 + (rot_i - ti) ** 2 + 1e-9).sum(dim=-1)

    def score_tail_batch(self, h, r, candidates):
        phase = self.rel_phase(r)
        rr = torch.cos(phase)
        ri = torch.sin(phase)
        hr = self.ent_re(h)
        hi = self.ent_im(h)
        rot_r = hr * rr - hi * ri
        rot_i = hr * ri + hi * rr
        tr = self.ent_re(candidates)
        ti = self.ent_im(candidates)
        return -torch.sqrt(
            (rot_r[:, None, :] - tr[None, :, :]) ** 2
            + (rot_i[:, None, :] - ti[None, :, :]) ** 2
            + 1e-9
        ).sum(dim=-1)


def make_model(name, num_entities, num_relations, dim):
    if name == "TransE":
        return TransE(num_entities, num_relations, dim)
    if name == "DistMult":
        return DistMult(num_entities, num_relations, dim)
    if name == "ComplEx":
        return ComplEx(num_entities, num_relations, dim)
    if name == "RotatE":
        return RotatE(num_entities, num_relations, dim)
    raise ValueError(name)


def perturb_relation_low(train_triples, budget, orig_degree, orig_relfreq, smoothing_c, seed):
    n = len(train_triples)
    target = int(n * budget)
    current_degree = dict(orig_degree)
    rel = np.array([orig_relfreq[r] for _, r, _ in train_triples], dtype=float)
    weights = 1.0 / (rel + smoothing_c)
    rng = np.random.default_rng(seed)
    keys = -np.log(rng.random(n)) / weights
    order = np.argsort(keys)
    removed = set()
    skipped = 0
    would_isolate = 0
    for idx in order:
        if len(removed) >= target:
            break
        h, _, t = train_triples[idx]
        if h == t:
            if current_degree[h] <= 2:
                skipped += 1
                would_isolate += 1
                continue
            current_degree[h] -= 2
            removed.add(int(idx))
            continue
        if current_degree[h] <= 1 or current_degree[t] <= 1:
            skipped += 1
            would_isolate += int(current_degree[h] <= 1) + int(current_degree[t] <= 1)
            continue
        current_degree[h] -= 1
        current_degree[t] -= 1
        removed.add(int(idx))
    if len(removed) != target:
        raise RuntimeError(f"Relation-low could remove only {len(removed)} / {target}")
    zero_degree = [entity for entity, degree in current_degree.items() if degree == 0]
    if zero_degree:
        raise AssertionError(f"Relation-low isolated training entities: {zero_degree[:10]}")
    kept = [x for i, x in enumerate(train_triples) if i not in removed]
    return kept, {
        "condition": "Relation-low",
        "requested_removed": target,
        "removed": len(removed),
        "skipped_candidate_deletions": skipped,
        "would_have_isolated_entities_count": would_isolate,
        "zero_degree_entities_after": 0,
        "weight_rule": "1/(original_relation_frequency+5)",
    }


def define_slices(test_triples, orig_degree, orig_relfreq):
    cov_scores = np.asarray([min(orig_degree[h], orig_degree[t]) for h, _, t in test_triples], dtype=float)
    rel_scores = np.asarray([orig_relfreq[r] for _, r, _ in test_triples], dtype=float)
    cq25, cq75 = np.quantile(cov_scores, [0.25, 0.75])
    rq25, rq75 = np.quantile(rel_scores, [0.25, 0.75])
    coverage_bins = ["low" if x <= cq25 else ("high" if x >= cq75 else "mid") for x in cov_scores]
    relation_bins = ["low" if x <= rq25 else ("high" if x >= rq75 else "mid") for x in rel_scores]
    return coverage_bins, relation_bins, {
        "source": "original_training_graph_before_perturbation",
        "coverage_statistic": "min(original_degree(head),original_degree(tail))",
        "relation_statistic": "original_training_relation_frequency",
        "coverage_q25_q75": [float(cq25), float(cq75)],
        "relation_q25_q75": [float(rq25), float(rq75)],
        "coverage_counts": {name: coverage_bins.count(name) for name in ("low", "mid", "high")},
        "relation_counts": {name: relation_bins.count(name) for name in ("low", "mid", "high")},
    }


def evaluate_rr(
    model,
    test_triples,
    filter_triples,
    num_entities,
    coverage_bins,
    relation_bins,
    device,
    query_batch_size=32,
    candidate_chunk_size=4096,
):
    """Compute optimistic-tie filtered tail ranks for every test triple."""
    model.eval()
    ftails = defaultdict(set)
    for h, r, t in filter_triples:
        ftails[(h, r)].add(t)
    rr = np.empty(len(test_triples), dtype=np.float64)
    with torch.no_grad():
        for q_start in range(0, len(test_triples), query_batch_size):
            batch = test_triples[q_start : q_start + query_batch_size]
            h = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            r = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
            t = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)
            target_scores = model.score(h, r, t)
            greater = torch.zeros(len(batch), dtype=torch.long, device=device)
            for c_start in range(0, num_entities, candidate_chunk_size):
                c_end = min(num_entities, c_start + candidate_chunk_size)
                candidates = torch.arange(c_start, c_end, dtype=torch.long, device=device)
                comparison = model.score_tail_batch(h, r, candidates) > target_scores[:, None]
                for row_idx, (hh, rr_id, tt) in enumerate(batch):
                    if c_start <= tt < c_end:
                        comparison[row_idx, tt - c_start] = False
                    for filtered_tail in ftails[(hh, rr_id)]:
                        if filtered_tail != tt and c_start <= filtered_tail < c_end:
                            comparison[row_idx, filtered_tail - c_start] = False
                greater += comparison.sum(dim=1)
            ranks = 1 + greater.detach().cpu().numpy()
            rr[q_start : q_start + len(batch)] = 1.0 / ranks

    def sliced_mean(labels, value):
        mask = np.asarray([x == value for x in labels], dtype=bool)
        return float(rr[mask].mean()) if mask.any() else None

    return {
        "overall_mrr": float(rr.mean()),
        "low_coverage_mrr": sliced_mean(coverage_bins, "low"),
        "mid_coverage_mrr": sliced_mean(coverage_bins, "mid"),
        "high_coverage_mrr": sliced_mean(coverage_bins, "high"),
        "low_relation_mrr": sliced_mean(relation_bins, "low"),
        "mid_relation_mrr": sliced_mean(relation_bins, "mid"),
        "high_relation_mrr": sliced_mean(relation_bins, "high"),
        "rr": rr.tolist(),
        "coverage_bins": coverage_bins,
        "relation_bins": relation_bins,
    }


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def progress(log_path, message):
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")
        f.flush()
        os.fsync(f.fileno())
    print(message, flush=True)


def run(args):
    cfg = {
        "seed": 11,
        "dataset": {
            "root": f"data/{args.dataset}", "train": "train.txt", "valid": "valid.txt", "test": "test.txt",
            "allow_fallback": False,
        },
        "model": {"name": args.model, "embedding_dim": 64},
        "training": {"epochs": 8, "batch_size": 512, "lr": 0.01, "negatives_per_positive": 5, "checkpoint_name": "checkpoint.pt"},
        "evaluation": {"test_queries": "all", "tail_only": True, "tie_handling": "optimistic: 1 + count(scores > target_score)"},
        "perturbation": {"budget": 0.30, "smoothing_c": 5},
        "runtime": {"device": "cpu"},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.progress_log) if args.progress_log else out_path.with_suffix(".progress.log")
    ds = load_dataset(cfg, str(out_path.with_suffix(".mapping.json")))
    filter_triples = ds["train"] + ds["valid"] + ds["test"]
    coverage_bins, relation_bins, slice_metadata = define_slices(ds["test"], ds["orig_degree"], ds["orig_relfreq"])
    device = torch.device("cpu")
    expected_metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "budget": 0.30,
        "embedding_dim": 64,
        "epochs": 8,
        "num_test_queries": len(ds["test"]),
        "train_seeds": TRAIN_SEEDS,
        "deletion_seeds": DELETION_SEEDS,
        "conditions": CONDITIONS,
    }
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        for key, value in expected_metadata.items():
            if result.get(key) != value:
                raise RuntimeError(f"Refusing incompatible resume: {key}={result.get(key)!r}, expected {value!r}")
    else:
        result = {
            **expected_metadata,
            "filter_membership": "all original train+valid+test triples; deleted training triples remain filters",
            "tie_handling": "optimistic: rank = 1 + number of unfiltered candidates with strictly greater score",
            "slice_metadata": slice_metadata,
            "runs": [],
            "complete": False,
        }
        atomic_json(out_path, result)
    completed = {(x["condition"], int(x["deletion_seed"]), int(x["train_seed"])) for x in result["runs"]}
    progress(log_path, f"START dataset={args.dataset} model={args.model} completed={len(completed)}/{EXPECTED_RUNS}")

    for condition in CONDITIONS:
        del_seeds = [0] if condition == "Original" else DELETION_SEEDS
        for deletion_seed in del_seeds:
            if condition == "Original":
                train_cond = ds["train"]
                perturb_meta = {"condition": "Original", "removed": 0, "zero_degree_entities_after": 0}
            elif condition in ("Random", "Structured-low"):
                train_cond, perturb_meta = perturb_training(
                    ds["train"], condition, 0.30, ds["orig_degree"], 5, deletion_seed,
                    valid_triples=ds["valid"], test_triples=ds["test"]
                )
            elif condition == "Relation-low":
                train_cond, perturb_meta = perturb_relation_low(
                    ds["train"], 0.30, ds["orig_degree"], ds["orig_relfreq"], 5, deletion_seed
                )
            else:
                raise ValueError(condition)
            for train_seed in TRAIN_SEEDS:
                run_key = (condition, deletion_seed, train_seed)
                if run_key in completed:
                    progress(log_path, f"SKIP completed condition={condition} deletion_seed={deletion_seed} train_seed={train_seed}")
                    continue
                start = time.time()
                progress(log_path, f"RUN condition={condition} deletion_seed={deletion_seed} train_seed={train_seed}")
                set_seed(train_seed)
                model = make_model(args.model, ds["num_entities"], ds["num_relations"], 64)
                losses = train_kge(model, train_cond, ds["num_entities"], cfg, device)
                metrics = evaluate_rr(
                    model, ds["test"], filter_triples, ds["num_entities"], coverage_bins, relation_bins, device
                )
                result["runs"].append({
                    "condition": condition,
                    "deletion_seed": deletion_seed,
                    "train_seed": train_seed,
                    "metrics": metrics,
                    "perturbation": perturb_meta,
                    "loss_start": float(losses[0]),
                    "loss_end": float(losses[-1]),
                    "runtime_sec": time.time() - start,
                })
                completed.add(run_key)
                result["complete"] = len(completed) == EXPECTED_RUNS
                atomic_json(out_path, result)
                progress(
                    log_path,
                    f"DONE condition={condition} deletion_seed={deletion_seed} train_seed={train_seed} "
                    f"overall_mrr={metrics['overall_mrr']:.8f} completed={len(completed)}/{EXPECTED_RUNS}",
                )
    if len(completed) != EXPECTED_RUNS:
        raise RuntimeError(f"Incomplete chunk: {len(completed)} / {EXPECTED_RUNS} runs")
    result["complete"] = True
    atomic_json(out_path, result)
    progress(log_path, f"COMPLETE dataset={args.dataset} model={args.model} runs={len(completed)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["WN18RR", "CoDEx-M"], required=True)
    ap.add_argument("--model", choices=["TransE", "DistMult", "ComplEx", "RotatE"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress-log")
    run(ap.parse_args())
