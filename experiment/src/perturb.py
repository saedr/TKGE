import numpy as np


def perturb_training(train_triples, condition, budget, orig_degree, smoothing_c, seed):
    n = len(train_triples)
    target = int(n * budget)
    degrees = dict(orig_degree)
    cov = np.array([min(orig_degree[h], orig_degree[t]) for h, _, t in train_triples], dtype=float)

    if condition == "Original":
        removed_idx = set()
        order = np.arange(n)
    else:
        if condition == "Random":
            w = np.ones(n)
        elif condition == "Structured-low":
            w = 1.0 / (cov + smoothing_c)
        elif condition == "Structured-high":
            w = cov + smoothing_c
        else:
            raise ValueError(condition)
        rng = np.random.default_rng(seed)
        keys = -np.log(rng.random(n)) / w
        order = np.argsort(keys)
        removed_idx = set()

    skipped = 0
    would_isolate = 0
    for idx in order:
        if len(removed_idx) >= target:
            break
        if condition == "Original":
            break
        h, _, t = train_triples[idx]
        if degrees[h] <= 1 or degrees[t] <= 1:
            skipped += 1
            would_isolate += int(degrees[h] <= 1) + int(degrees[t] <= 1)
            continue
        degrees[h] -= 1
        degrees[t] -= 1
        removed_idx.add(int(idx))

    kept = [trip for i, trip in enumerate(train_triples) if i not in removed_idx]
    deg_vals = np.array(list(degrees.values()), dtype=float)
    zero_deg = int((deg_vals == 0).sum())
    return kept, {
        "condition": condition,
        "requested_removed": target,
        "removed": len(removed_idx),
        "skipped_candidate_deletions": skipped,
        "would_have_isolated_entities_count": would_isolate,
        "final_degree_min": float(deg_vals.min()),
        "final_degree_max": float(deg_vals.max()),
        "final_degree_mean": float(deg_vals.mean()),
        "zero_degree_entities_after": zero_deg,
    }
