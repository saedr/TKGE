#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_SHA = "5049b84501657abc44c36dd7d2db5f8519e5ee8f"
DATA_REPO = "stasl0217/UKGE"
UNKGCP_SHA = "6cdf658be1825be0b369084859dff129ae3c113f"
TAU = 0.85
CP_LEVEL = 0.90
RNG_WORLD = 20260908

DATASETS = ["nl27k", "cn15k"]

UKGE_CFG = {
    "dim": 128,
    "batch_size": 4096,
    "neg_per_pos": 2,
    "lr": 1e-3,
    "max_epochs": 200,
    "check_every": 5,
    "patience_checks": 8,
    "seed": 0,
    "psl_weight": 0.2,
}

DOWN_CFG = {
    "dim": 64,
    "batch_size": 4096,
    "neg_per_pos": 1,
    "lr": 1e-3,
    "epochs": 10,
    "world_seed": 11,
    "seed_controls": [11, 23, 37, 53, 71],
}


def log(msg):
    print(msg, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_dataset(dataset, root):
    root = Path(root) / dataset
    root.mkdir(parents=True, exist_ok=True)
    files = ["train.tsv", "val.tsv", "test.tsv", "softlogic.tsv"]
    base = f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_SHA}/data/{dataset}"
    for name in files:
        out = root / name
        if not out.exists():
            url = f"{base}/{name}"
            log(f"download {url}")
            urllib.request.urlretrieve(url, out)
    return root


def read_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4:
                continue
            rows.append((row[0], row[1], row[2], float(row[3])))
    return rows


def build_mapping(*splits):
    ents = sorted({x for split in splits for h, r, t, w in split for x in (h, t)})
    rels = sorted({r for split in splits for h, r, t, w in split})
    e2i = {e: i for i, e in enumerate(ents)}
    r2i = {r: i for i, r in enumerate(rels)}
    return e2i, r2i


def encode(rows, e2i, r2i):
    arr = np.empty((len(rows), 4), dtype=np.float64)
    for i, (h, r, t, w) in enumerate(rows):
        arr[i] = (e2i[h], r2i[r], e2i[t], w)
    return arr


class UKGELogi(nn.Module):
    def __init__(self, nent, nrel, dim):
        super().__init__()
        self.ent = nn.Embedding(nent, dim)
        self.rel = nn.Embedding(nrel, dim)
        with torch.no_grad():
            nn.init.trunc_normal_(self.ent.weight, mean=0.0, std=0.3, a=-0.6, b=0.6)
            nn.init.trunc_normal_(self.rel.weight, mean=0.0, std=0.3, a=-0.6, b=0.6)
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def raw(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(dim=-1)

    def score(self, h, r, t):
        return torch.sigmoid(self.scale * self.raw(h, r, t) + self.bias)


class ComplEx(nn.Module):
    def __init__(self, nent, nrel, dim):
        super().__init__()
        self.er = nn.Embedding(nent, dim)
        self.ei = nn.Embedding(nent, dim)
        self.rr = nn.Embedding(nrel, dim)
        self.ri = nn.Embedding(nrel, dim)
        for emb in (self.er, self.ei, self.rr, self.ri):
            nn.init.xavier_uniform_(emb.weight)

    def score(self, h, r, t):
        hr, hi = self.er(h), self.ei(h)
        rr, ri = self.rr(r), self.ri(r)
        tr, ti = self.er(t), self.ei(t)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)

    def all_tail_scores(self, h, r):
        hr, hi = self.er(h), self.ei(h)
        rr, ri = self.rr(r), self.ri(r)
        qr = hr * rr - hi * ri
        qi = hi * rr + hr * ri
        return qr @ self.er.weight.T + qi @ self.ei.weight.T


def rejection_negatives(h, r, t, nent, k, known, rng, corrupt_head):
    b = len(h)
    out = np.empty((b, k), dtype=np.int64)
    for i in range(b):
        for j in range(k):
            for _ in range(100):
                x = int(rng.integers(nent))
                cand = (x, int(r[i]), int(t[i])) if corrupt_head else (int(h[i]), int(r[i]), x)
                if cand not in known:
                    out[i, j] = x
                    break
            else:
                out[i, j] = x
    return out


def predict_ukge(model, arr, device="cpu", batch=16384):
    model.eval()
    vals = []
    with torch.no_grad():
        for i in range(0, len(arr), batch):
            x = arr[i:i + batch]
            h = torch.as_tensor(x[:, 0], dtype=torch.long, device=device)
            r = torch.as_tensor(x[:, 1], dtype=torch.long, device=device)
            t = torch.as_tensor(x[:, 2], dtype=torch.long, device=device)
            vals.append(model.score(h, r, t).cpu().numpy())
    return np.concatenate(vals) if vals else np.array([], dtype=float)


def train_ukge(train, val, soft, nent, nrel, known, cfg, device):
    set_seed(cfg["seed"])
    model = UKGELogi(nent, nrel, cfg["dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"])
    best = None
    best_mse = float("inf")
    stale = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        nb = 0
        for start in range(0, len(order), cfg["batch_size"]):
            ids = order[start:start + cfg["batch_size"]]
            x = train[ids]
            h_np = x[:, 0].astype(np.int64)
            r_np = x[:, 1].astype(np.int64)
            t_np = x[:, 2].astype(np.int64)
            w_np = x[:, 3].astype(np.float32)
            nh_np = rejection_negatives(h_np, r_np, t_np, nent, cfg["neg_per_pos"], known, rng, True)
            nt_np = rejection_negatives(h_np, r_np, t_np, nent, cfg["neg_per_pos"], known, rng, False)
            h = torch.as_tensor(h_np, dtype=torch.long, device=device)
            r = torch.as_tensor(r_np, dtype=torch.long, device=device)
            t = torch.as_tensor(t_np, dtype=torch.long, device=device)
            w = torch.as_tensor(w_np, dtype=torch.float32, device=device)
            nh = torch.as_tensor(nh_np, dtype=torch.long, device=device)
            nt = torch.as_tensor(nt_np, dtype=torch.long, device=device)

            pos = model.score(h, r, t)
            pos_loss = (pos - w).square().mean()
            rr = r[:, None].expand_as(nh)
            tt = t[:, None].expand_as(nh)
            hh = h[:, None].expand_as(nt)
            neg_h = model.score(nh, rr, tt)
            neg_t = model.score(hh, rr, nt)
            neg_loss = 0.5 * (neg_h.square().mean() + neg_t.square().mean())
            loss = pos_loss + neg_loss

            if len(soft):
                srow = soft[int(rng.integers(len(soft)))]
                sh = torch.tensor([int(srow[0])], dtype=torch.long, device=device)
                sr = torch.tensor([int(srow[1])], dtype=torch.long, device=device)
                st = torch.tensor([int(srow[2])], dtype=torch.long, device=device)
                sw = torch.tensor([float(srow[3])], dtype=torch.float32, device=device)
                sp = model.score(sh, sr, st)
                loss = loss + cfg["psl_weight"] * F.relu(sw - sp).square().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1

        if epoch % cfg["check_every"] == 0 or epoch == cfg["max_epochs"]:
            vp = predict_ukge(model, val, device)
            mse = float(np.mean((vp - val[:, 3]) ** 2))
            log(f"UKGE epoch={epoch} train_loss={total / max(nb, 1):.6f} val_mse={mse:.6f}")
            if mse < best_mse - 1e-6:
                best_mse = mse
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg["patience_checks"]:
                    break
    if best is not None:
        model.load_state_dict(best)
    return model, {"best_val_mse": best_mse, "epochs_ran": epoch}


def entropy_uncertainty(scores, dataset):
    s = np.asarray(scores, dtype=float)
    if dataset == "cn15k":
        s = 0.5 * s + 0.5
    s = np.clip(s, 1e-6, 1 - 1e-6)
    u = -s * np.log(s) - (1 - s) * np.log(1 - s)
    return np.clip(u, 1e-6, None)


def higher_quantile(x, q):
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0:
        return float("nan")
    idx = int(math.ceil(q * (len(x) - 1)))
    idx = min(max(idx, 0), len(x) - 1)
    return float(x[idx])


def conformal_intervals(val_scores, val_y, test_scores, dataset, level=0.90):
    u_val = entropy_uncertainty(val_scores, dataset)
    nc = np.abs(val_scores - val_y) / u_val
    qlevel = math.ceil((len(nc) + 1) * level) / len(nc)
    qlevel = min(qlevel, 1.0)
    qhat = higher_quantile(nc, qlevel)
    u_test = entropy_uncertainty(test_scores, dataset)
    lo = np.maximum(0.0, test_scores - qhat * u_test)
    hi = np.minimum(1.0, test_scores + qhat * u_test)
    return lo, hi, qhat


def train_complex(graph, nent, nrel, seed, cfg, device):
    set_seed(seed)
    model = ComplEx(nent, nrel, cfg["dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    arr = np.asarray(graph, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(len(arr))
        for start in range(0, len(order), cfg["batch_size"]):
            x = arr[order[start:start + cfg["batch_size"]]]
            h = torch.as_tensor(x[:, 0], dtype=torch.long, device=device)
            r = torch.as_tensor(x[:, 1], dtype=torch.long, device=device)
            t = torch.as_tensor(x[:, 2], dtype=torch.long, device=device)
            pos = model.score(h, r, t)
            nh = h.repeat_interleave(cfg["neg_per_pos"])
            nr = r.repeat_interleave(cfg["neg_per_pos"])
            nt = torch.as_tensor(rng.integers(0, nent, size=len(nh)), dtype=torch.long, device=device)
            neg = model.score(nh, nr, nt)
            loss = F.softplus(-pos).mean() + F.softplus(neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def query_outputs(model, queries, nent, device, batch=64):
    winners = []
    margins = []
    all_scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(queries), batch):
            x = queries[start:start + batch]
            h = torch.as_tensor(x[:, 0], dtype=torch.long, device=device)
            r = torch.as_tensor(x[:, 1], dtype=torch.long, device=device)
            scores = model.all_tail_scores(h, r)
            vals, idx = torch.topk(scores, 2, dim=1)
            winners.extend(idx[:, 0].cpu().tolist())
            margins.extend((vals[:, 0] - vals[:, 1]).cpu().tolist())
            all_scores.append(scores.cpu())
    return np.asarray(winners), np.asarray(margins), torch.cat(all_scores, dim=0)


def filtered_metrics(scores, queries, known_strong):
    s = scores.numpy().copy()
    ranks = []
    hits10 = []
    tails_by_hr = defaultdict(set)
    for h, r, t in known_strong:
        tails_by_hr[(int(h), int(r))].add(int(t))
    for i, row in enumerate(queries):
        h, r, t = int(row[0]), int(row[1]), int(row[2])
        true_score = s[i, t]
        for other in tails_by_hr.get((h, r), ()):
            if other != t:
                s[i, other] = -np.inf
        rank = 1 + int(np.sum(s[i] > true_score))
        ranks.append(rank)
        hits10.append(rank <= 10)
    mrr = float(np.mean([1.0 / r for r in ranks]))
    return mrr, float(np.mean(hits10)), ranks


def pair_disagree(winner_arrays):
    vals = []
    for a, b in combinations(winner_arrays, 2):
        vals.append(float(np.mean(a != b)))
    return float(np.mean(vals)) if vals else 0.0


def entropy_from_winners(winner_arrays):
    mat = np.stack(winner_arrays, axis=0)
    out = []
    for j in range(mat.shape[1]):
        c = Counter(mat[:, j].tolist())
        p = np.array(list(c.values()), dtype=float) / mat.shape[0]
        out.append(float(-(p * np.log(p)).sum()))
    return out


def run_dataset(dataset, data_root, out_root, device):
    root = download_dataset(dataset, data_root)
    train_s = read_tsv(root / "train.tsv")
    val_s = read_tsv(root / "val.tsv")
    test_s = read_tsv(root / "test.tsv")
    soft_s = read_tsv(root / "softlogic.tsv")
    e2i, r2i = build_mapping(train_s, val_s, test_s, soft_s)
    train = encode(train_s, e2i, r2i)
    val = encode(val_s, e2i, r2i)
    test = encode(test_s, e2i, r2i)
    soft = encode(soft_s, e2i, r2i)
    nent = len(e2i)
    nrel = len(r2i)
    known = {(int(x[0]), int(x[1]), int(x[2])) for x in np.concatenate([train, val, test, soft], axis=0)}
    log(f"{dataset}: entities={nent} relations={nrel} train={len(train)} val={len(val)} test={len(test)}")

    ukge, ukge_meta = train_ukge(train, val, soft, nent, nrel, known, UKGE_CFG, device)
    val_scores = predict_ukge(ukge, val, device)

    order = np.lexsort((test[:, 2], test[:, 0], test[:, 1]))
    test_sorted = test[order]
    U = test_sorted[::2]
    Q = test_sorted[1::2]
    u_scores = predict_ukge(ukge, U, device)
    lo, hi, qhat = conformal_intervals(val_scores, val[:, 3], u_scores, dataset, CP_LEVEL)
    coverage = float(np.mean((lo <= U[:, 3]) & (U[:, 3] <= hi)))
    lengths = hi - lo
    defstrong = lo > TAU
    defweak = hi <= TAU
    ambiguous = (lo <= TAU) & (TAU < hi)
    amb_count = int(ambiguous.sum())
    amb_frac = float(ambiguous.mean())
    strong_Q = Q[Q[:, 3] > TAU][:500]
    q_available = int(np.sum(Q[:, 3] > TAU))
    feasibility = {
        "coverage": coverage,
        "avg_interval_length": float(lengths.mean()),
        "qhat": float(qhat),
        "ambiguous_count": amb_count,
        "ambiguous_fraction": amb_frac,
        "strong_queries_available": q_available,
        "passed": bool(coverage >= 0.88 and amb_count >= 100 and 0.05 <= amb_frac <= 0.80 and q_available >= 500),
    }
    log(f"{dataset} feasibility {feasibility}")
    base = [(int(x[0]), int(x[1]), int(x[2])) for x in train if x[3] > TAU]
    ds = [(int(U[i, 0]), int(U[i, 1]), int(U[i, 2])) for i in np.where(defstrong)[0]]
    amb = [(int(U[i, 0]), int(U[i, 1]), int(U[i, 2])) for i in np.where(ambiguous)[0]]
    nominal_mask = u_scores[ambiguous] > TAU

    result = {
        "dataset": dataset,
        "source": {"ukge_data_repo": DATA_REPO, "ukge_data_sha": DATA_SHA, "unkgcp_sha": UNKGCP_SHA},
        "counts": {"entities": nent, "relations": nrel, "train": len(train), "val": len(val), "test": len(test)},
        "ukge": ukge_meta,
        "feasibility": feasibility,
    }
    outd = Path(out_root) / dataset
    outd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outd / "intervals.npz", lo=lo, hi=hi, scores=u_scores, truth=U[:, 3], ambiguous=ambiguous)

    if not feasibility["passed"]:
        result["decision"] = "KILL-BENCHMARK"
        with open(outd / "RESULT.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    rng = np.random.default_rng(RNG_WORLD)
    masks = {
        "LOWER": np.zeros(len(amb), dtype=bool),
        "UPPER": np.ones(len(amb), dtype=bool),
        "NOMINAL": np.asarray(nominal_mask, dtype=bool),
    }
    for j in range(9):
        masks[f"RANDOM_{j + 1}"] = rng.random(len(amb)) < 0.5

    fixed = list(dict.fromkeys(base + ds))
    world_winners = {}
    world_margins = {}
    nominal_scores = None
    for name, mask in masks.items():
        graph = fixed + [amb[i] for i in np.where(mask)[0]]
        graph = list(dict.fromkeys(graph))
        log(f"{dataset} train world={name} edges={len(graph)}")
        model = train_complex(graph, nent, nrel, DOWN_CFG["world_seed"], DOWN_CFG, device)
        win, margin, scores = query_outputs(model, strong_Q, nent, device)
        world_winners[name] = win
        world_margins[name] = margin
        if name == "NOMINAL":
            nominal_scores = scores

    world_arrays = [world_winners[name] for name in masks.keys()]
    world_stack = np.stack(world_arrays, axis=0)
    witness = np.asarray([len(set(world_stack[:, i].tolist())) > 1 for i in range(len(strong_Q))], dtype=bool)
    witness_rate = float(witness.mean())
    world_pair = pair_disagree(world_arrays)

    seed_arrays = [world_winners["NOMINAL"]]
    nominal_graph = fixed + [amb[i] for i in np.where(masks["NOMINAL"])[0]]
    nominal_graph = list(dict.fromkeys(nominal_graph))
    for seed in DOWN_CFG["seed_controls"][1:]:
        log(f"{dataset} train seed-control={seed}")
        model = train_complex(nominal_graph, nent, nrel, seed, DOWN_CFG, device)
        win, _, _ = query_outputs(model, strong_Q, nent, device)
        seed_arrays.append(win)
    seed_pair = pair_disagree(seed_arrays)
    excess = world_pair - seed_pair

    nm = world_margins["NOMINAL"]
    cutoff = float(np.quantile(nm, 0.75, method="linear"))
    high = nm >= cutoff
    high_witness = float(witness[high].mean()) if high.any() else 0.0

    all_strong = [(int(x[0]), int(x[1]), int(x[2])) for x in np.concatenate([train, val, test], axis=0) if x[3] > TAU]
    mrr, h10, ranks = filtered_metrics(nominal_scores, strong_Q, all_strong)
    ent = entropy_from_winners(world_arrays)

    gate = {
        "WitnessRate": witness_rate,
        "WorldPairDisagree": world_pair,
        "SeedPairDisagree": seed_pair,
        "ExcessDisagree": excess,
        "HighMarginWitnessRate": high_witness,
        "NominalMRR": mrr,
        "NominalHits10": h10,
        "passed": bool(witness_rate >= 0.25 and world_pair >= 0.10 and excess >= 0.05 and high_witness >= 0.10 and h10 >= 0.20),
    }
    result["phenomenon"] = gate
    result["diagnostics"] = {
        "high_margin_cutoff": cutoff,
        "mean_top1_entropy": float(np.mean(ent)),
        "max_top1_entropy": float(np.max(ent)),
        "correctness_change_fraction": float(np.mean([
            len(set((world_stack[:, i] == int(strong_Q[i, 2])).tolist())) > 1 for i in range(len(strong_Q))
        ])),
    }
    rel_stats = {}
    for r in np.unique(strong_Q[:, 1].astype(int)):
        ix = np.where(strong_Q[:, 1].astype(int) == r)[0]
        if len(ix) >= 20:
            rel_stats[str(int(r))] = {"n": int(len(ix)), "witness_rate": float(witness[ix].mean())}
    result["relation_diagnostics"] = rel_stats
    result["decision"] = "CONTINUE" if gate["passed"] else "KILL-PHENOMENON"
    with open(outd / "RESULT.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(outd / "query_diagnostics.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["query_index", "h", "r", "t", "truth_conf", "witness", "nominal_margin", "top1_entropy", "nominal_rank"])
        for i, row in enumerate(strong_Q):
            wr.writerow([i, int(row[0]), int(row[1]), int(row[2]), float(row[3]), int(witness[i]), float(nm[i]), float(ent[i]), int(ranks[i])])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="experiment/identified_ranking/data")
    ap.add_argument("--out-root", default="experiment/identified_ranking/results")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")
    results = []
    for ds in DATASETS:
        results.append(run_dataset(ds, args.data_root, args.out_root, device))
    if any(r["decision"] == "KILL-BENCHMARK" for r in results):
        decision = "KILL-BENCHMARK"
    elif all(r["decision"] == "CONTINUE" for r in results):
        decision = "CONTINUE"
    else:
        decision = "KILL-PHENOMENON"
    summary = {
        "decision": decision,
        "datasets": results,
        "frozen_thresholds": {
            "coverage_min": 0.88,
            "ambiguous_min": 100,
            "ambiguous_fraction": [0.05, 0.80],
            "strong_queries_min": 500,
            "WitnessRate": 0.25,
            "WorldPairDisagree": 0.10,
            "ExcessDisagree": 0.05,
            "HighMarginWitnessRate": 0.10,
            "NominalHits10": 0.20,
        },
    }
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_root) / "RESULT.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
