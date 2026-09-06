#!/usr/bin/env python3
"""Frozen DistMult-only prospective reliability gate on EMERGE.

Scientific question: does a relation-normalized DistMult signal, computed only
from the Jan-1 KG snapshot, improve prediction of near-term EMERGE Deprecate
versus Exists outcomes beyond the pre-outcome controls frozen in the preceding
feasibility gate?

No model family, target, feature family, or threshold is tuned after results.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from experiment.scripts.prospective_revision_feasibility import (
    TARGET_YEARS,
    YEARS,
    cohort_summary,
    collect_year,
    download_inputs,
    enrich,
    historical_relation_rates,
    rows_to_df,
    scan_snapshot,
)

# Frozen before observing KGE results.
DIM = 32
EPOCHS = 2
BATCH_SIZE = 8192
BLOCK_SIZE = 500_000
NEGATIVES = 1
LR = 0.10
SEED = 1701
BOOTSTRAP_REPS = 4000
DELTA_AUPRC_GATE = 0.005


class SparseDistMult(torch.nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int):
        super().__init__()
        self.entity = torch.nn.Embedding(n_entities, dim, sparse=True)
        self.relation = torch.nn.Embedding(n_relations, dim, sparse=True)
        bound = math.sqrt(6.0 / dim)
        torch.nn.init.uniform_(self.entity.weight, -bound, bound)
        torch.nn.init.uniform_(self.relation.weight, -bound, bound)

    def score(self, h, r, t):
        return (self.entity(h) * self.relation(r) * self.entity(t)).sum(dim=-1)


def encode_snapshot(path: Path, binary_path: Path):
    """Encode a gzip TSV snapshot into an int32 memmap-friendly binary file."""
    entity_to_id: dict[bytes, int] = {}
    relation_to_id: dict[bytes, int] = {}
    n_rows = 0
    malformed = 0
    buffer = []
    flush_rows = 200_000
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    def entity_id(x: bytes):
        value = entity_to_id.get(x)
        if value is None:
            value = len(entity_to_id)
            entity_to_id[x] = value
        return value

    def relation_id(x: bytes):
        value = relation_to_id.get(x)
        if value is None:
            value = len(relation_to_id)
            relation_to_id[x] = value
        return value

    with open(binary_path, "wb") as out, gzip.open(path, "rb") as f:
        for line in f:
            parts = line.rstrip(b"\r\n").split(b"\t")
            if len(parts) < 3:
                malformed += 1
                continue
            h, r, t = parts[0], parts[1], parts[2]
            buffer.append((entity_id(h), relation_id(r), entity_id(t)))
            if len(buffer) >= flush_rows:
                arr = np.asarray(buffer, dtype=np.int32)
                out.write(arr.tobytes(order="C"))
                n_rows += len(arr)
                buffer.clear()
            if (n_rows + len(buffer)) and (n_rows + len(buffer)) % 5_000_000 == 0:
                print(f"ENCODE {path.name}: rows={n_rows + len(buffer):,} entities={len(entity_to_id):,} relations={len(relation_to_id):,}", flush=True)
        if buffer:
            arr = np.asarray(buffer, dtype=np.int32)
            out.write(arr.tobytes(order="C"))
            n_rows += len(arr)
            buffer.clear()
    if n_rows <= 0:
        raise RuntimeError(f"No triples encoded from {path}")
    triples = np.memmap(binary_path, dtype=np.int32, mode="r", shape=(n_rows, 3))
    return triples, entity_to_id, relation_to_id, {"n_triples": n_rows, "malformed": malformed}


def train_distmult(triples, n_entities, n_relations, year, checkpoint_dir: Path):
    torch.manual_seed(SEED + year)
    np.random.seed(SEED + year)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")
    model = SparseDistMult(n_entities, n_relations, DIM).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)
    n = len(triples)
    n_blocks = math.ceil(n / BLOCK_SIZE)
    losses = []
    rng = np.random.default_rng(SEED + year)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        block_order = rng.permutation(n_blocks)
        epoch_loss = 0.0
        n_batches = 0
        for block_pos, block_id in enumerate(block_order, start=1):
            start = int(block_id) * BLOCK_SIZE
            end = min(n, start + BLOCK_SIZE)
            block = np.asarray(triples[start:end], dtype=np.int64).copy()
            rng.shuffle(block, axis=0)
            for j in range(0, len(block), BATCH_SIZE):
                batch = torch.from_numpy(block[j:j + BATCH_SIZE]).to(device)
                h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
                pos = model.score(h, r, t)
                nh = h.repeat_interleave(NEGATIVES)
                nr = r.repeat_interleave(NEGATIVES)
                nt = torch.randint(0, n_entities, (len(h) * NEGATIVES,), device=device)
                neg = model.score(nh, nr, nt)
                loss = torch.nn.functional.softplus(-pos).mean() + torch.nn.functional.softplus(neg).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            if block_pos % 10 == 0 or block_pos == n_blocks:
                print(f"TRAIN {year} epoch={epoch + 1}/{EPOCHS} blocks={block_pos}/{n_blocks} mean_loss={epoch_loss/max(1,n_batches):.6f}", flush=True)
        mean_loss = epoch_loss / max(1, n_batches)
        losses.append(mean_loss)
        # Durable per-epoch checkpoint inside the job artifact path.
        torch.save({
            "year": year,
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "losses": losses,
            "config": frozen_config(),
        }, checkpoint_dir / f"distmult_{year}_epoch{epoch + 1}.pt")
    return model, losses


def separation_sanity(model, triples, n_entities, year, n_sample=50_000):
    rng = np.random.default_rng(SEED + 99 + year)
    ids = rng.integers(0, len(triples), size=min(n_sample, len(triples)))
    arr = np.asarray(triples[ids], dtype=np.int64)
    with torch.no_grad():
        h = torch.from_numpy(arr[:, 0])
        r = torch.from_numpy(arr[:, 1])
        t = torch.from_numpy(arr[:, 2])
        nt = torch.from_numpy(rng.integers(0, n_entities, size=len(arr), dtype=np.int64))
        pos = model.score(h, r, t).cpu().numpy()
        neg = model.score(h, r, nt).cpu().numpy()
    return {
        "positive_mean": float(pos.mean()),
        "random_negative_mean": float(neg.mean()),
        "positive_gt_random_negative": float((pos > neg).mean()),
    }


def relation_score_stats(model, triples, n_relations, year):
    """Compute relation score mean/std on the entire Jan-1 snapshot only."""
    counts = np.zeros(n_relations, dtype=np.int64)
    sums = np.zeros(n_relations, dtype=np.float64)
    sums2 = np.zeros(n_relations, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(triples), BLOCK_SIZE):
            end = min(len(triples), start + BLOCK_SIZE)
            block = np.asarray(triples[start:end], dtype=np.int64)
            for j in range(0, len(block), 32768):
                batch = torch.from_numpy(block[j:j + 32768])
                h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
                scores = model.score(h, r, t).cpu().numpy().astype(np.float64)
                rel = batch[:, 1].cpu().numpy()
                counts += np.bincount(rel, minlength=n_relations)
                sums += np.bincount(rel, weights=scores, minlength=n_relations)
                sums2 += np.bincount(rel, weights=scores * scores, minlength=n_relations)
            if end % 5_000_000 < BLOCK_SIZE or end == len(triples):
                print(f"NORMALIZE {year}: scored={end:,}/{len(triples):,}", flush=True)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    second = np.divide(sums2, counts, out=np.zeros_like(sums2), where=counts > 0)
    var = np.maximum(second - means * means, 1e-12)
    std = np.sqrt(var)
    return means, std, counts


def attach_kge_scores(rows, model, entity_to_id, relation_to_id, rel_mean, rel_std):
    eligible = [dict(r) for r in rows if r["in_snapshot"]]
    missing = 0
    raw_scores = []
    encoded = []
    kept = []
    for row in eligible:
        h = entity_to_id.get(row["h"].encode())
        r = relation_to_id.get(row["r"].encode())
        t = entity_to_id.get(row["t"].encode())
        if h is None or r is None or t is None:
            missing += 1
            continue
        encoded.append((h, r, t))
        kept.append(row)
    if missing:
        raise RuntimeError(f"{missing} cohort triples verified in snapshot but absent from encoder mapping")
    arr = np.asarray(encoded, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(arr), 32768):
            batch = torch.from_numpy(arr[start:start + 32768])
            score = model.score(batch[:, 0], batch[:, 1], batch[:, 2]).cpu().numpy()
            raw_scores.extend(score.tolist())
    for row, raw, enc in zip(kept, raw_scores, encoded):
        rid = enc[1]
        row["distmult_score_raw"] = float(raw)
        row["distmult_score_relation_z"] = float((raw - rel_mean[rid]) / rel_std[rid])
    return kept


def make_model(include_kge: bool):
    numeric = [
        "log_head_degree", "log_tail_degree", "log_relation_frequency",
        "log_age_days", "age_missing", "history_deprecation_rate",
    ]
    if include_kge:
        numeric.append("distmult_score_relation_z")
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["r"]),
    ])
    return Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
    ])


def fit_predictions(train_rows, test_rows, include_kge):
    train = rows_to_df(train_rows)
    test = rows_to_df(test_rows)
    # rows_to_df preserves extra numeric columns.
    y_train = train["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)
    model = make_model(include_kge)
    model.fit(train, y_train)
    pred = model.predict_proba(test)[:, 1]
    return y_test, pred


def paired_bootstrap(y, baseline, augmented, reps=BOOTSTRAP_REPS):
    rng = np.random.default_rng(SEED)
    n = len(y)
    deltas = []
    base_vals = []
    aug_vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.sum() == 0:
            continue
        b = average_precision_score(yy, baseline[idx])
        a = average_precision_score(yy, augmented[idx])
        base_vals.append(b)
        aug_vals.append(a)
        deltas.append(a - b)
    if not deltas:
        raise RuntimeError("No valid bootstrap replicates")
    q = np.quantile(deltas, [0.025, 0.5, 0.975])
    return {
        "replicates_requested": reps,
        "replicates_used": len(deltas),
        "delta_ci_low": float(q[0]),
        "delta_median": float(q[1]),
        "delta_ci_high": float(q[2]),
        "baseline_bootstrap_mean": float(np.mean(base_vals)),
        "augmented_bootstrap_mean": float(np.mean(aug_vals)),
    }


def frozen_config():
    return {
        "model": "DistMult",
        "embedding_dim": DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "negative_sampling": "uniform tail corruption",
        "negatives_per_positive": NEGATIVES,
        "optimizer": "torch.optim.Adagrad sparse embeddings",
        "learning_rate": LR,
        "seed_rule": "1701 + snapshot year",
        "relation_normalization": "z-score against all triples of the same relation in that Jan-1 snapshot",
        "primary_metric": "AUPRC on 2025 eligible cohort",
        "gate": "delta AUPRC >= 0.005 and paired-bootstrap 95% CI for delta excludes zero above 0",
        "bootstrap_replications": BOOTSTRAP_REPS,
    }


def write_rows(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    workdir = Path(args.workdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outdir / "checkpoints"
    paths = download_inputs(workdir)

    cohorts = {}
    source_stats = {}
    for year in YEARS:
        cohorts[year], source_stats[year] = collect_year(year, paths["corpus"][year])
        print(f"COHORT {year}: {len(cohorts[year]):,}", flush=True)
    hist_2024, _ = historical_relation_rates(cohorts, [2023])
    hist_2025, _ = historical_relation_rates(cohorts, [2023, 2024])

    enriched = {}
    cohort_summaries = {}
    for year, hist in ((2024, hist_2024), (2025, hist_2025)):
        print(f"VERIFY/CONTROLS {year}: scanning snapshot", flush=True)
        scan = scan_snapshot(paths["kg"][year], cohorts[year])
        enriched[year], _ = enrich(year, cohorts[year], scan, hist)
        # Frozen risk set: facts that actually existed in the Jan-1 snapshot.
        enriched[year] = [row for row in enriched[year] if row["in_snapshot"]]
        cohort_summaries[year] = cohort_summary(enriched[year])
        print(f"ELIGIBLE {year}: n={len(enriched[year]):,} positives={sum(x['label'] for x in enriched[year]):,}", flush=True)

    kge_rows = {}
    training = {}
    for year in TARGET_YEARS:
        bin_path = workdir / f"encoded_{year}.int32"
        print(f"ENCODING full snapshot {year}", flush=True)
        triples, ent_map, rel_map, enc_stats = encode_snapshot(paths["kg"][year], bin_path)
        print(f"TRAINING DistMult {year}: triples={len(triples):,} entities={len(ent_map):,} relations={len(rel_map):,}", flush=True)
        model, losses = train_distmult(triples, len(ent_map), len(rel_map), year, checkpoint_dir)
        sanity = separation_sanity(model, triples, len(ent_map), year)
        print(f"SANITY {year}: losses={losses} separation={sanity}", flush=True)
        rel_mean, rel_std, rel_counts = relation_score_stats(model, triples, len(rel_map), year)
        kge_rows[year] = attach_kge_scores(enriched[year], model, ent_map, rel_map, rel_mean, rel_std)
        training[year] = {
            "encoding": {**enc_stats, "n_entities": len(ent_map), "n_relations": len(rel_map)},
            "losses": [float(x) for x in losses],
            "loss_decreased": bool(losses[-1] < losses[0]),
            "separation_sanity": sanity,
            "relations_scored": int((rel_counts > 0).sum()),
        }
        write_rows(outdir / f"cohort_{year}_with_distmult.csv", kge_rows[year])
        del model, triples, ent_map, rel_map, rel_mean, rel_std, rel_counts
        gc.collect()
        try:
            bin_path.unlink()
        except FileNotFoundError:
            pass

    y, pred_base = fit_predictions(kge_rows[2024], kge_rows[2025], include_kge=False)
    y2, pred_aug = fit_predictions(kge_rows[2024], kge_rows[2025], include_kge=True)
    if not np.array_equal(y, y2):
        raise AssertionError("Baseline and augmented test labels differ")
    base_ap = float(average_precision_score(y, pred_base))
    aug_ap = float(average_precision_score(y, pred_aug))
    delta_ap = aug_ap - base_ap
    base_auc = float(roc_auc_score(y, pred_base))
    aug_auc = float(roc_auc_score(y, pred_aug))
    boot = paired_bootstrap(y, pred_base, pred_aug)
    passes = bool(delta_ap >= DELTA_AUPRC_GATE and boot["delta_ci_low"] > 0.0)
    training_sane = all(training[y0]["loss_decreased"] and training[y0]["separation_sanity"]["positive_gt_random_negative"] > 0.5 for y0 in TARGET_YEARS)
    decision = "PASS" if passes and training_sane else ("TRAINING_SANITY_FAILURE" if not training_sane else "KILL")

    summary = {
        "decision": decision,
        "frozen_config": frozen_config(),
        "cohorts": cohort_summaries,
        "training": training,
        "test_2025": {
            "n": int(len(y)),
            "positives": int(y.sum()),
            "prevalence": float(y.mean()),
            "controls_auprc": base_ap,
            "controls_plus_distmult_auprc": aug_ap,
            "delta_auprc": delta_ap,
            "controls_auroc": base_auc,
            "controls_plus_distmult_auroc": aug_auc,
            "paired_bootstrap": boot,
            "gate_threshold": DELTA_AUPRC_GATE,
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Prospective KGE reliability pilot — DistMult only

## Decision: **{decision}**

Frozen scientific gate: add the relation-normalized DistMult signal to the full controls-only model, then require **ΔAUPRC ≥ {DELTA_AUPRC_GATE:.3f}** on the untouched 2025 cohort and a paired-bootstrap 95% CI for ΔAUPRC entirely above zero.

## 2025 primary result

| Quantity | Value |
|---|---:|
| Eligible triples | {len(y):,} |
| Deprecate positives | {int(y.sum()):,} |
| Positive prevalence | {y.mean():.6f} |
| Controls AUPRC | {base_ap:.6f} |
| Controls + DistMult AUPRC | {aug_ap:.6f} |
| **ΔAUPRC** | **{delta_ap:.6f}** |
| Paired-bootstrap 95% CI for Δ | [{boot['delta_ci_low']:.6f}, {boot['delta_ci_high']:.6f}] |
| Controls AUROC | {base_auc:.6f} |
| Controls + DistMult AUROC | {aug_auc:.6f} |

## Training sanity

| Snapshot | First epoch loss | Final epoch loss | Positive > random-negative rate | Sanity |
|---|---:|---:|---:|---|
| 2024 | {training[2024]['losses'][0]:.6f} | {training[2024]['losses'][-1]:.6f} | {training[2024]['separation_sanity']['positive_gt_random_negative']:.4f} | {training[2024]['loss_decreased'] and training[2024]['separation_sanity']['positive_gt_random_negative'] > 0.5} |
| 2025 | {training[2025]['losses'][0]:.6f} | {training[2025]['losses'][-1]:.6f} | {training[2025]['separation_sanity']['positive_gt_random_negative']:.4f} | {training[2025]['loss_decreased'] and training[2025]['separation_sanity']['positive_gt_random_negative'] > 0.5} |

## Frozen KGE configuration

- DistMult only.
- 32-dimensional embeddings.
- Two full-snapshot epochs per year.
- One uniformly sampled tail negative per positive.
- Sparse Adagrad, learning rate 0.10.
- Identical settings for 2024 and 2025.
- Relation normalization uses the score distribution of **all triples of that relation in the Jan-1 snapshot**, never the future-labeled cohort.
- The risk set includes only triples verified to exist in the corresponding Jan-1 snapshot.
- The controls model is unchanged from the feasibility gate: relation identity, head/tail degree, relation frequency, triple age/missingness, and historical relation deprecation rate.
- Bootstrap resamples 2025 test triples with paired baseline/augmented predictions; models are not refit inside each bootstrap replicate.

If the decision is `KILL`, the frozen plan forbids adding ComplEx, TransE, GNNs, extra targets, or hand-tuned features to rescue the result.
"""
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--outdir", required=True)
    main(ap.parse_args())
