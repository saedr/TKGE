#!/usr/bin/env python3
import argparse
import json
import time
from collections import defaultdict
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
    kept = [x for i, x in enumerate(train_triples) if i not in removed]
    return kept, {
        "condition": "Relation-low",
        "requested_removed": target,
        "removed": len(removed),
        "skipped_candidate_deletions": skipped,
        "would_have_isolated_entities_count": would_isolate,
        "zero_degree_entities_after": int(sum(v == 0 for v in current_degree.values())),
        "weight_rule": "1/(original_relation_frequency+5)",
    }


def evaluate_rr(model, test_triples, filter_triples, num_entities, orig_degree, orig_relfreq, max_queries, device):
    model.eval()
    ftails = defaultdict(set)
    for h, r, t in filter_triples:
        ftails[(h, r)].add(t)
    subset = test_triples[:max_queries]
    cov_scores = np.array([min(orig_degree[h], orig_degree[t]) for h, _, t in subset], dtype=float)
    rel_scores = np.array([orig_relfreq[r] for _, r, _ in subset], dtype=float)
    cq25, cq75 = np.quantile(cov_scores, [0.25, 0.75])
    rq25, rq75 = np.quantile(rel_scores, [0.25, 0.75])
    rr = []
    cov_bins = []
    rel_bins = []
    with torch.no_grad():
        cand_t = torch.arange(num_entities, device=device)
        for h, r, t in subset:
            h_t = torch.full((num_entities,), h, device=device, dtype=torch.long)
            r_t = torch.full((num_entities,), r, device=device, dtype=torch.long)
            scores = model.score(h_t, r_t, cand_t).detach().cpu().numpy()
            for tt in ftails[(h, r)]:
                if tt != t:
                    scores[tt] = -1e12
            rank = 1 + int((scores > scores[t]).sum())
            rr.append(1.0 / rank)
            cov = min(orig_degree[h], orig_degree[t])
            relf = orig_relfreq[r]
            cov_bins.append("low" if cov <= cq25 else ("high" if cov >= cq75 else "mid"))
            rel_bins.append("low-frequency" if relf <= rq25 else ("high-frequency" if relf >= rq75 else "mid-frequency"))
    rr = np.asarray(rr, dtype=float)
    def mean_mask(labels, value):
        mask = np.array([x == value for x in labels], dtype=bool)
        return float(rr[mask].mean()) if mask.any() else None
    return {
        "overall_mrr": float(rr.mean()),
        "low_coverage_mrr": mean_mask(cov_bins, "low"),
        "mid_coverage_mrr": mean_mask(cov_bins, "mid"),
        "high_coverage_mrr": mean_mask(cov_bins, "high"),
        "low_relation_mrr": mean_mask(rel_bins, "low-frequency"),
        "mid_relation_mrr": mean_mask(rel_bins, "mid-frequency"),
        "high_relation_mrr": mean_mask(rel_bins, "high-frequency"),
        "rr": rr.tolist(),
        "coverage_bins": cov_bins,
        "relation_bins": rel_bins,
    }


def run(args):
    cfg = {
        "seed": 11,
        "dataset": {"root": f"data/{args.dataset}", "train": "train.txt", "valid": "valid.txt", "test": "test.txt"},
        "model": {"name": args.model, "embedding_dim": 64},
        "training": {"epochs": 8, "batch_size": 512, "lr": 0.01, "negatives_per_positive": 5, "checkpoint_name": "checkpoint.pt"},
        "evaluation": {"max_test_queries": 400, "topk": 10},
        "perturbation": {"budget": 0.30, "smoothing_c": 5},
        "runtime": {"device": "cpu"},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(cfg, str(Path(args.out).with_suffix(".mapping.json")))
    filter_triples = ds["train"] + ds["valid"] + ds["test"]
    device = torch.device("cpu")
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "budget": 0.30,
        "embedding_dim": 64,
        "epochs": 8,
        "max_test_queries": 400,
        "train_seeds": TRAIN_SEEDS,
        "deletion_seeds": DELETION_SEEDS,
        "runs": [],
    }
    for condition in CONDITIONS:
        del_seeds = [0] if condition == "Original" else DELETION_SEEDS
        for deletion_seed in del_seeds:
            if condition == "Original":
                train_cond = ds["train"]
                perturb_meta = {"condition": "Original", "removed": 0}
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
                start = time.time()
                set_seed(train_seed)
                model = make_model(args.model, ds["num_entities"], ds["num_relations"], 64)
                losses = train_kge(model, train_cond, ds["num_entities"], cfg, device)
                metrics = evaluate_rr(
                    model, ds["test"], filter_triples, ds["num_entities"], ds["orig_degree"], ds["orig_relfreq"], 400, device
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
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(result, f)
                print(args.dataset, args.model, condition, deletion_seed, train_seed, metrics["overall_mrr"], flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["WN18RR", "CoDEx-M"], required=True)
    ap.add_argument("--model", choices=["TransE", "DistMult", "ComplEx", "RotatE"], required=True)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
