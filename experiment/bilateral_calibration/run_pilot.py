#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MASTER_SEED = 20260908
DOWNSTREAM_SEEDS = [11, 23, 37]
NOISE_RATE = 0.05
BOOTSTRAPS = 2000


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


class TransE(torch.nn.Module):
    def __init__(self, n_ent, n_rel, dim=64):
        super().__init__()
        self.ent = torch.nn.Embedding(n_ent, dim)
        self.rel = torch.nn.Embedding(n_rel, dim)
        torch.nn.init.xavier_uniform_(self.ent.weight)
        torch.nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h, r, t):
        return -(self.ent(h) + self.rel(r) - self.ent(t)).norm(p=2, dim=-1)


class ComplEx(torch.nn.Module):
    def __init__(self, n_ent, n_rel, dim=64):
        super().__init__()
        self.ent_re = torch.nn.Embedding(n_ent, dim)
        self.ent_im = torch.nn.Embedding(n_ent, dim)
        self.rel_re = torch.nn.Embedding(n_rel, dim)
        self.rel_im = torch.nn.Embedding(n_rel, dim)
        for emb in [self.ent_re, self.ent_im, self.rel_re, self.rel_im]:
            torch.nn.init.xavier_uniform_(emb.weight)

    def score(self, h, r, t):
        hr, hi = self.ent_re(h), self.ent_im(h)
        rr, ri = self.rel_re(r), self.rel_im(r)
        tr, ti = self.ent_re(t), self.ent_im(t)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                rows.append(tuple(p))
    return rows


def load_dataset(ds_dir):
    train_s = read_tsv(ds_dir / "train.txt")
    valid_s = read_tsv(ds_dir / "dev.txt")
    test_s = read_tsv(ds_dir / "test.txt")
    ents = sorted({x for triples in [train_s, valid_s, test_s] for h, _, t in triples for x in (h, t)})
    rels = sorted({r for triples in [train_s, valid_s, test_s] for _, r, _ in triples})
    e2i = {e: i for i, e in enumerate(ents)}
    r2i = {r: i for i, r in enumerate(rels)}
    def enc(rows):
        return [(e2i[h], r2i[r], e2i[t]) for h, r, t in rows]
    train, valid, test = enc(train_s), enc(valid_s), enc(test_s)
    ref = set(train) | set(valid) | set(test)
    return {
        "train": train, "valid": valid, "test": test,
        "ref": ref, "n_ent": len(ents), "n_rel": len(rels),
        "entities": ents, "relations": rels,
    }


def relation_domains(train):
    heads, tails = defaultdict(set), defaultdict(set)
    for h, r, t in train:
        heads[r].add(h); tails[r].add(t)
    return {r: sorted(v) for r, v in heads.items()}, {r: sorted(v) for r, v in tails.items()}


def filtered_negatives(batch, n_ent, forbidden, k, rng):
    out = []
    for h, r, t in batch:
        for _ in range(k):
            side = int(rng.integers(0, 2))
            for _try in range(100):
                x = int(rng.integers(0, n_ent))
                q = (x, r, t) if side == 0 else (h, r, x)
                if q not in forbidden:
                    out.append(q)
                    break
            else:
                for x in range(n_ent):
                    q = (x, r, t) if side == 0 else (h, r, x)
                    if q not in forbidden:
                        out.append(q); break
    return np.asarray(out, dtype=np.int64)


def train_transe(train, n_ent, n_rel, ref, seed, epochs=20, batch_size=4096):
    seed_all(seed)
    model = TransE(n_ent, n_rel, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    arr = np.asarray(train, dtype=np.int64)
    rng = np.random.default_rng(seed)
    losses = []
    for ep in range(epochs):
        perm = rng.permutation(len(arr))
        total = 0.0; nb = 0
        for st in range(0, len(arr), batch_size):
            b = arr[perm[st:st+batch_size]]
            neg = filtered_negatives(b, n_ent, ref, 1, rng)
            bt = torch.from_numpy(b)
            nt = torch.from_numpy(neg)
            ps = model.score(bt[:,0], bt[:,1], bt[:,2])
            ns = model.score(nt[:,0], nt[:,1], nt[:,2])
            loss = F.relu(1.0 - ps + ns).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); nb += 1
        losses.append(total / max(nb, 1))
        if ep in {0, 4, 9, 14, 19}:
            print(f"generator epoch={ep+1}/{epochs} loss={losses[-1]:.5f}", flush=True)
    return model, losses


def score_triples(model, triples, batch_size=65536):
    if not triples:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(triples, dtype=np.int64)
    vals = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(arr), batch_size):
            x = torch.from_numpy(arr[st:st+batch_size])
            vals.append(model.score(x[:,0], x[:,1], x[:,2]).cpu().numpy())
    return np.concatenate(vals).astype(np.float64)


def fixed_negative_for(triple, head_dom, tail_dom, ref, n_ent, rng):
    h, r, t = triple
    sides = [int(rng.integers(0, 2))]
    sides.append(1 - sides[0])
    for side in sides:
        pool = head_dom.get(r, []) if side == 0 else tail_dom.get(r, [])
        if pool:
            start = int(rng.integers(0, len(pool)))
            for j in range(len(pool)):
                x = pool[(start+j) % len(pool)]
                q = (x, r, t) if side == 0 else (h, r, x)
                if q not in ref:
                    return q
    start = int(rng.integers(0, n_ent))
    side = sides[0]
    for j in range(n_ent):
        x = (start+j) % n_ent
        q = (x, r, t) if side == 0 else (h, r, x)
        if q not in ref:
            return q
    raise RuntimeError("Could not generate fixed negative")


def generator_pairwise_accuracy(model, test, head_dom, tail_dom, ref, n_ent):
    rng = np.random.default_rng(MASTER_SEED + 100)
    idx = rng.permutation(len(test))[:min(5000, len(test))]
    pos = [test[i] for i in idx]
    neg = [fixed_negative_for(q, head_dom, tail_dom, ref, n_ent, rng) for q in pos]
    ps = score_triples(model, pos); ns = score_triples(model, neg)
    return float(np.mean(ps > ns))


def hard_false_additions(model, source, target, head_dom, tail_dom, ref, rng):
    order = rng.permutation(len(source))
    out = []
    used = set()
    model.eval()
    for oi in order:
        h, r, t = source[int(oi)]
        first = int(rng.integers(0, 2))
        made = None
        for side in [first, 1-first]:
            pool = head_dom.get(r, []) if side == 0 else tail_dom.get(r, [])
            cand = []
            for x in pool:
                q = (x, r, t) if side == 0 else (h, r, x)
                if q != (h, r, t) and q not in ref and q not in used:
                    cand.append(q)
            if not cand:
                continue
            sc = score_triples(model, cand)
            best_score = float(np.max(sc))
            best_ids = np.flatnonzero(sc == best_score)
            if len(best_ids) == 1:
                made = cand[int(best_ids[0])]
            else:
                made = min(cand[int(i)] for i in best_ids)
            break
        if made is not None:
            out.append(made); used.add(made)
        if len(out) >= target:
            break
    return out


def train_complex(train, n_ent, n_rel, ref, extra_forbidden, seed, epochs=30, batch_size=4096, neg_k=4):
    seed_all(seed)
    model = ComplEx(n_ent, n_rel, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    arr = np.asarray(train, dtype=np.int64)
    rng = np.random.default_rng(seed)
    forbidden = set(ref) | set(extra_forbidden)
    losses = []
    for ep in range(epochs):
        perm = rng.permutation(len(arr))
        total = 0.0; nb = 0
        for st in range(0, len(arr), batch_size):
            b = arr[perm[st:st+batch_size]]
            neg = filtered_negatives(b, n_ent, forbidden, neg_k, rng)
            bt = torch.from_numpy(b); nt = torch.from_numpy(neg)
            ps = model.score(bt[:,0], bt[:,1], bt[:,2])
            ns = model.score(nt[:,0], nt[:,1], nt[:,2])
            loss = F.softplus(-ps).mean() + F.softplus(ns).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); nb += 1
        losses.append(total / max(nb, 1))
        if ep in {0, 4, 9, 19, 29}:
            print(f"complex seed={seed} epoch={ep+1}/{epochs} loss={losses[-1]:.5f}", flush=True)
    return model, losses


def make_clean_pool(positives, head_dom, tail_dom, ref, n_ent, seed):
    rng = np.random.default_rng(seed)
    triples = []; obs = []; refy = []; groups = []
    for p in positives:
        n = fixed_negative_for(p, head_dom, tail_dom, ref, n_ent, rng)
        base = len(triples)
        triples += [p, n]; obs += [1, 0]; refy += [1, 0]; groups.append((base, base+1))
    return triples, np.asarray(obs, np.int8), np.asarray(refy, np.int8), groups


def make_noisy_pool(positives, additions, head_dom, tail_dom, ref, n_ent, seed):
    triples, obs, refy, groups = make_clean_pool(positives, head_dom, tail_dom, ref, n_ent, seed)
    triples.extend(additions)
    obs = np.concatenate([obs, np.ones(len(additions), dtype=np.int8)])
    refy = np.concatenate([refy, np.zeros(len(additions), dtype=np.int8)])
    return triples, obs, refy, groups


def fit_platt(scores, y):
    s = torch.tensor(np.asarray(scores), dtype=torch.float64)
    yy = torch.tensor(np.asarray(y), dtype=torch.float64)
    a = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    opt = torch.optim.LBFGS([a,b], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        z = a*s + b
        loss = F.binary_cross_entropy_with_logits(z, yy) + 1e-6*(a*a+b*b)
        loss.backward()
        return loss
    opt.step(closure)
    return float(a.detach()), float(b.detach())


def probs_from(scores, ab):
    a,b = ab
    z = np.clip(a*np.asarray(scores, dtype=np.float64)+b, -40, 40)
    return 1.0/(1.0+np.exp(-z))


def ece10(p, y):
    p = np.asarray(p); y = np.asarray(y)
    if len(p) == 0: return 0.0
    e = 0.0
    for i in range(10):
        lo, hi = i/10, (i+1)/10
        m = (p >= lo) & ((p < hi) if i < 9 else (p <= hi))
        if np.any(m):
            e += float(np.mean(m)) * abs(float(np.mean(p[m])) - float(np.mean(y[m])))
    return float(e)


def auc(p, y):
    p = np.asarray(p, dtype=np.float64); y = np.asarray(y, dtype=np.int8)
    pos = int(y.sum()); neg = len(y)-pos
    if pos == 0 or neg == 0: return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(p):
        j = i+1
        while j < len(p) and p[order[j]] == p[order[i]]: j += 1
        ranks[order[i:j]] = (i+1 + j)/2.0
        i = j
    s = float(ranks[y==1].sum())
    return (s - pos*(pos+1)/2.0)/(pos*neg)


def metrics(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1-1e-7)
    y = np.asarray(y, dtype=np.float64)
    out = {
        "ece10": ece10(p,y),
        "brier": float(np.mean((p-y)**2)),
        "nll": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
        "auroc": auc(p,y.astype(np.int8)),
    }
    try:
        logit = np.log(p/(1-p))
        aa,bb = fit_platt(logit, y)
        out["calibration_slope"] = aa
        out["calibration_intercept"] = bb
    except Exception:
        out["calibration_slope"] = None; out["calibration_intercept"] = None
    return out


def evaluate_model(model, pools, calibrator_pool_key):
    scores = {k: score_triples(model, v[0]) for k,v in pools.items()}
    fit_labels = pools[calibrator_pool_key][1]
    ab = fit_platt(scores[calibrator_pool_key], fit_labels)
    probs = {k: probs_from(v, ab) for k,v in scores.items()}
    return ab, probs


def bootstrap_bex(test_groups, seed_records):
    rng = np.random.default_rng(MASTER_SEED)
    ng = len(test_groups)
    vals = []
    for _ in range(BOOTSTRAPS):
        gs = rng.integers(0, ng, size=ng)
        idx = np.empty(2*ng, dtype=np.int64)
        for j,g in enumerate(gs):
            a,b = test_groups[int(g)]
            idx[2*j] = a; idx[2*j+1] = b
        ds = []
        for rec in seed_records:
            y = rec["test_y"][idx]
            ds.append(ece10(rec["bilateral_test_p"][idx], y) - ece10(rec["label_test_p"][idx], y))
        vals.append(float(np.mean(ds)))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def write_additions(path, rows, entities, relations):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for h,r,t in rows:
            w.writerow([entities[h], relations[r], entities[t]])


def run_dataset(name, ds_dir, out_dir):
    print(f"=== DATASET {name} ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_dataset(ds_dir)
    train, valid, test, ref = d["train"], d["valid"], d["test"], d["ref"]
    hd, td = relation_domains(train)

    rng_dev = np.random.default_rng(MASTER_SEED)
    perm = rng_dev.permutation(len(valid))
    cut = len(valid)//2
    dev_fit = [valid[int(i)] for i in perm[:cut]]
    dev_eval = [valid[int(i)] for i in perm[cut:]]

    generator, gen_losses = train_transe(train, d["n_ent"], d["n_rel"], ref, MASTER_SEED)
    gen_acc = generator_pairwise_accuracy(generator, test, hd, td, ref, d["n_ent"])
    print(f"{name} generator_pairwise_accuracy={gen_acc:.4f}", flush=True)

    def target(n): return int(math.ceil(NOISE_RATE/(1-NOISE_RATE)*n))
    rng_add = np.random.default_rng(MASTER_SEED)
    add_train = hard_false_additions(generator, train, target(len(train)), hd, td, ref, rng_add)
    add_fit = hard_false_additions(generator, dev_fit, target(len(dev_fit)), hd, td, ref, rng_add)
    add_eval = hard_false_additions(generator, dev_eval, target(len(dev_eval)), hd, td, ref, rng_add)
    print(f"{name} additions train={len(add_train)}/{target(len(train))} fit={len(add_fit)}/{target(len(dev_fit))} eval={len(add_eval)}/{target(len(dev_eval))}", flush=True)
    write_additions(out_dir/"false_train.tsv", add_train, d["entities"], d["relations"])
    write_additions(out_dir/"false_dev_fit.tsv", add_fit, d["entities"], d["relations"])
    write_additions(out_dir/"false_dev_eval.tsv", add_eval, d["entities"], d["relations"])

    clean_fit = make_clean_pool(dev_fit, hd, td, ref, d["n_ent"], MASTER_SEED+201)
    clean_eval = make_clean_pool(dev_eval, hd, td, ref, d["n_ent"], MASTER_SEED+202)
    noisy_fit = make_noisy_pool(dev_fit, add_fit, hd, td, ref, d["n_ent"], MASTER_SEED+201)
    noisy_eval = make_noisy_pool(dev_eval, add_eval, hd, td, ref, d["n_ent"], MASTER_SEED+202)
    test_pool = make_clean_pool(test, hd, td, ref, d["n_ent"], MASTER_SEED+203)

    pools_clean = {"fit": clean_fit, "eval": clean_eval, "test": test_pool}
    pools_noisy = {"fit": noisy_fit, "eval": noisy_eval, "test": test_pool}

    seed_summaries = []
    bootstrap_records = []
    for seed in DOWNSTREAM_SEEDS:
        print(f"--- {name} downstream seed {seed} clean ---", flush=True)
        clean_model, clean_losses = train_complex(train, d["n_ent"], d["n_rel"], ref, [], seed)
        clean_ab, clean_p = evaluate_model(clean_model, pools_clean, "fit")
        label_ab, label_p = evaluate_model(clean_model, pools_noisy, "fit")

        print(f"--- {name} downstream seed {seed} bilateral ---", flush=True)
        bilat_train = list(train) + list(add_train)
        bilat_model, bilat_losses = train_complex(bilat_train, d["n_ent"], d["n_rel"], ref, add_train, seed)
        bilat_ab, bilat_p = evaluate_model(bilat_model, pools_noisy, "fit")

        clean_eval_m = metrics(clean_p["eval"], clean_eval[1])
        clean_test_m = metrics(clean_p["test"], test_pool[2])
        label_eval_obs = metrics(label_p["eval"], noisy_eval[1])
        label_eval_ref = metrics(label_p["eval"], noisy_eval[2])
        label_test = metrics(label_p["test"], test_pool[2])
        bilat_eval_obs = metrics(bilat_p["eval"], noisy_eval[1])
        bilat_eval_ref = metrics(bilat_p["eval"], noisy_eval[2])
        bilat_test = metrics(bilat_p["test"], test_pool[2])
        bex = bilat_test["ece10"] - label_test["ece10"]
        bex_brier = bilat_test["brier"] - label_test["brier"]
        lgap = label_test["ece10"] - clean_test_m["ece10"]
        auc_drop = label_test["auroc"] - bilat_test["auroc"]
        rec = {
            "seed": seed,
            "clean_calibrator": {"a": clean_ab[0], "b": clean_ab[1]},
            "label_only_calibrator": {"a": label_ab[0], "b": label_ab[1]},
            "bilateral_calibrator": {"a": bilat_ab[0], "b": bilat_ab[1]},
            "clean_dev_observed": clean_eval_m,
            "clean_test_reference": clean_test_m,
            "label_dev_observed": label_eval_obs,
            "label_dev_reference": label_eval_ref,
            "label_test_reference": label_test,
            "bilateral_dev_observed": bilat_eval_obs,
            "bilateral_dev_reference": bilat_eval_ref,
            "bilateral_test_reference": bilat_test,
            "bex_test": bex,
            "bex_brier": bex_brier,
            "lgap_test": lgap,
            "auroc_drop": auc_drop,
            "clean_loss_final": clean_losses[-1],
            "bilateral_loss_final": bilat_losses[-1],
        }
        print(f"{name} seed={seed} clean_ece={clean_test_m['ece10']:.4f} label_ece={label_test['ece10']:.4f} bilat_ece={bilat_test['ece10']:.4f} BEX={bex:.4f} auc_drop={auc_drop:.4f}", flush=True)
        seed_summaries.append(rec)
        bootstrap_records.append({
            "test_y": test_pool[2],
            "label_test_p": label_p["test"],
            "bilateral_test_p": bilat_p["test"],
        })

    ci = bootstrap_bex(test_pool[3], bootstrap_records)
    bex_vals = [r["bex_test"] for r in seed_summaries]
    brier_vals = [r["bex_brier"] for r in seed_summaries]
    auc_drops = [r["auroc_drop"] for r in seed_summaries]
    targets = {"train": target(len(train)), "dev_fit": target(len(dev_fit)), "dev_eval": target(len(dev_eval))}
    actuals = {"train": len(add_train), "dev_fit": len(add_fit), "dev_eval": len(add_eval)}
    summary = {
        "dataset": name,
        "counts": {"entities": d["n_ent"], "relations": d["n_rel"], "train": len(train), "dev": len(valid), "test": len(test)},
        "generator_pairwise_accuracy": gen_acc,
        "false_additions_target": targets,
        "false_additions_actual": actuals,
        "mean_bex_test": float(np.mean(bex_vals)),
        "bex_test_values": bex_vals,
        "bex_bootstrap_95": ci,
        "mean_bex_brier": float(np.mean(brier_vals)),
        "bex_brier_values": brier_vals,
        "mean_auroc_drop": float(np.mean(auc_drops)),
        "seed_results": seed_summaries,
        "generator_loss_final": gen_losses[-1],
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def decide(summaries):
    feasible = True; reasons = []
    for s in summaries:
        if s["generator_pairwise_accuracy"] < .65:
            feasible = False; reasons.append(f"{s['dataset']}: generator accuracy")
        for k,v in s["false_additions_target"].items():
            if s["false_additions_actual"][k] < .98*v:
                feasible = False; reasons.append(f"{s['dataset']}: additions {k}")
        for r in s["seed_results"]:
            if r["clean_test_reference"]["auroc"] < .70:
                feasible = False; reasons.append(f"{s['dataset']} seed {r['seed']}: clean AUROC")
            if r["clean_dev_observed"]["ece10"] > .05:
                feasible = False; reasons.append(f"{s['dataset']} seed {r['seed']}: clean dev ECE")
    if not feasible:
        return "KILL-BENCHMARK", reasons

    cal_ok = True; reasons = []
    for s in summaries:
        for r in s["seed_results"]:
            if r["label_dev_observed"]["ece10"] > .05:
                cal_ok = False; reasons.append(f"{s['dataset']} seed {r['seed']}: label-only observed ECE")
            if r["bilateral_dev_observed"]["ece10"] > .05:
                cal_ok = False; reasons.append(f"{s['dataset']} seed {r['seed']}: bilateral observed ECE")
    if not cal_ok:
        return "KILL-CALIBRATION-FEASIBILITY", reasons

    ph_ok = True; reasons = []
    for s in summaries:
        if s["mean_bex_test"] < .03:
            ph_ok = False; reasons.append(f"{s['dataset']}: mean BEX < .03")
        if not all(x > 0 for x in s["bex_test_values"]):
            ph_ok = False; reasons.append(f"{s['dataset']}: BEX not positive all seeds")
        if s["bex_bootstrap_95"][0] <= .01:
            ph_ok = False; reasons.append(f"{s['dataset']}: BEX bootstrap lower <= .01")
        if s["mean_bex_brier"] < .005:
            ph_ok = False; reasons.append(f"{s['dataset']}: mean Brier excess < .005")
        if not all(x > 0 for x in s["bex_brier_values"]):
            ph_ok = False; reasons.append(f"{s['dataset']}: Brier excess not positive all seeds")
    if not ph_ok:
        return "KILL-PHENOMENON", reasons

    specificity = all(s["mean_auroc_drop"] <= .05 for s in summaries)
    if not specificity:
        return "STOP-SPECIFICITY", [f"{s['dataset']}: AUROC drop > .05" for s in summaries if s["mean_auroc_drop"] > .05]
    return "CONTINUE", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    specs = [("FB15k-237", root/"fb15k-237"), ("WN18RR", root/"wn18rr")]
    summaries = []
    start = time.time()
    try:
        for name, path in specs:
            summaries.append(run_dataset(name, path, out/name))
        decision, reasons = decide(summaries)
        result = {
            "decision": decision,
            "reasons": reasons,
            "frozen_thresholds": {
                "noise_rate": .05,
                "generator_pairwise_accuracy": .65,
                "min_addition_fraction": .98,
                "clean_auroc": .70,
                "observed_ece": .05,
                "mean_bex_ece": .03,
                "all_seed_bex_positive": True,
                "bootstrap_lower": .01,
                "mean_brier_excess": .005,
                "all_seed_brier_positive": True,
                "max_auroc_drop": .05,
            },
            "datasets": summaries,
            "runtime_seconds": time.time()-start,
        }
    except Exception as e:
        result = {"decision": "KILL-BENCHMARK", "stage": "technical_execution", "error": repr(e), "runtime_seconds": time.time()-start}
        (out/"RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        raise
    (out/"RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("FINAL_DECISION=" + result["decision"], flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
