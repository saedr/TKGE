import numpy as np


def _incident_examples(train_triples, entity_id, removed_idx, limit=5):
    kept = []
    removed = []
    for i, (h, r, t) in enumerate(train_triples):
        if h != entity_id and t != entity_id:
            continue
        row = (int(h), int(r), int(t))
        if i in removed_idx:
            if len(removed) < limit:
                removed.append(row)
        elif len(kept) < limit:
            kept.append(row)
    return kept, removed


def perturb_training(train_triples, condition, budget, orig_degree, smoothing_c, seed, valid_triples=None, test_triples=None):
    n = len(train_triples)
    target = 0 if condition == "Original" else int(n * budget)
    current_degree = dict(orig_degree)
    cov = np.array([min(orig_degree[h], orig_degree[t]) for h, _, t in train_triples], dtype=float)
    train_entities = set(orig_degree.keys())
    valid_entities = set()
    test_entities = set()
    if valid_triples is not None:
        for h, _, t in valid_triples:
            valid_entities.add(h)
            valid_entities.add(t)
    if test_triples is not None:
        for h, _, t in test_triples:
            test_entities.add(h)
            test_entities.add(t)

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
        if h == t:
            # degree counts incident edge count; self-loop contributes twice
            decrement = 2
            if current_degree[h] <= decrement:
                skipped += 1
                would_isolate += 1
                continue
            current_degree[h] -= decrement
            removed_idx.add(int(idx))
            continue

        if current_degree[h] <= 1 or current_degree[t] <= 1:
            skipped += 1
            would_isolate += int(current_degree[h] <= 1) + int(current_degree[t] <= 1)
            continue
        current_degree[h] -= 1
        current_degree[t] -= 1
        removed_idx.add(int(idx))

    if len(removed_idx) != target:
        raise RuntimeError(
            f"Unable to satisfy deletion budget for condition={condition}. "
            f"requested_removed={target}, removed={len(removed_idx)}, skipped={skipped}."
        )

    zero_after = [e for e in train_entities if current_degree.get(e, 0) == 0]
    violations = []
    for entity_id in zero_after:
        kept_examples, removed_examples = _incident_examples(train_triples, entity_id, removed_idx)
        incident_total = sum(1 for h, _, t in train_triples if h == entity_id or t == entity_id)
        removed_incident = sum(1 for i, (h, _, t) in enumerate(train_triples) if i in removed_idx and (h == entity_id or t == entity_id))
        appears_in_valid_test_only = (entity_id not in train_entities) and (entity_id in valid_entities or entity_id in test_entities)
        violations.append(
            {
                "entity_id": int(entity_id),
                "original_degree": int(orig_degree.get(entity_id, 0)),
                "final_degree": int(current_degree.get(entity_id, 0)),
                "original_incident_triples": int(incident_total),
                "removed_incident_triples": int(removed_incident),
                "removed_incident_examples": removed_examples,
                "remaining_incident_examples": kept_examples,
                "appears_in_train": entity_id in train_entities,
                "appears_in_valid_or_test": entity_id in valid_entities or entity_id in test_entities,
                "appears_only_in_valid_test": appears_in_valid_test_only,
            }
        )
    assert not zero_after, f"zero-degree train-visible entities after perturbation: {violations}"

    kept = [trip for i, trip in enumerate(train_triples) if i not in removed_idx]
    deg_vals = np.array([current_degree[e] for e in train_entities], dtype=float)
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
        "zero_degree_violations": violations,
    }
