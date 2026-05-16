from collections import defaultdict
import numpy as np
import torch


def _qbins(values):
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return q25, q75


def evaluate_tail_only(model, test_triples, filter_triples, num_entities, orig_degree, orig_relfreq, max_queries, topk, device):
    model.eval()
    ftails = defaultdict(set)
    for h, r, t in filter_triples:
        ftails[(h, r)].add(t)

    subset = test_triples[: max_queries or len(test_triples)]
    cov_scores = np.array([min(orig_degree[h], orig_degree[t]) for h, _, t in subset], dtype=float)
    rel_scores = np.array([orig_relfreq[r] for _, r, _ in subset], dtype=float)
    cq25, cq75 = _qbins(cov_scores)
    rq25, rq75 = _qbins(rel_scores)

    rows = []
    top10_rows = []
    for query_idx, (h, r, t) in enumerate(subset):
        h_t = torch.full((num_entities,), h, device=device, dtype=torch.long)
        r_t = torch.full((num_entities,), r, device=device, dtype=torch.long)
        cand_t = torch.arange(num_entities, device=device)
        scores = model.score(h_t, r_t, cand_t).detach().cpu().numpy()
        for tt in ftails[(h, r)]:
            if tt != t:
                scores[tt] = -1e12
        rank = 1 + int((scores > scores[t]).sum())
        top10 = np.argsort(-scores)[:topk]
        cov = min(orig_degree[h], orig_degree[t])
        relf = orig_relfreq[r]
        cov_bin = "low" if cov <= cq25 else ("high" if cov >= cq75 else "mid")
        rel_bin = "low-frequency" if relf <= rq25 else ("high-frequency" if relf >= rq75 else "mid-frequency")
        rows.append((rank, cov_bin, rel_bin))
        top10_rows.append({
            "query_idx": int(query_idx),
            "h": int(h),
            "r": int(r),
            "t": int(t),
            "coverage_bin": cov_bin,
            "relation_frequency_bin": rel_bin,
            "top10_tail_ids": [int(x) for x in top10.tolist()],
        })

    ranks = np.array([r[0] for r in rows], dtype=float)
    mrr = float((1.0 / ranks).mean())
    h10 = float((ranks <= 10).mean())

    def group(ix, value):
        vals = [1.0 / x[0] if value == "mrr" else float(x[0] <= 10) for x in rows if x[ix] == ix_name]
        return float(np.mean(vals)) if vals else None

    cov_bins = ["low", "mid", "high"]
    rel_bins = ["low-frequency", "mid-frequency", "high-frequency"]
    by_cov = {}
    by_rel = {}
    for ix_name in cov_bins:
        vals_m = [1.0 / x[0] for x in rows if x[1] == ix_name]
        vals_h = [float(x[0] <= 10) for x in rows if x[1] == ix_name]
        by_cov[ix_name] = {"tail_mrr": float(np.mean(vals_m)) if vals_m else None, "tail_hits_at_10": float(np.mean(vals_h)) if vals_h else None}
    for ix_name in rel_bins:
        vals_m = [1.0 / x[0] for x in rows if x[2] == ix_name]
        vals_h = [float(x[0] <= 10) for x in rows if x[2] == ix_name]
        by_rel[ix_name] = {"tail_mrr": float(np.mean(vals_m)) if vals_m else None, "tail_hits_at_10": float(np.mean(vals_h)) if vals_h else None}

    return {
        "evaluation_kind": "tail_only",
        "shared_filtering": "original_train_valid_test",
        "tail_mrr": mrr,
        "tail_hits_at_10": h10,
        "coverage_bins_source": "original_train_graph",
        "relation_frequency_bins_source": "original_train_graph",
        "by_coverage_bin": by_cov,
        "by_relation_frequency_bin": by_rel,
        "num_queries": len(subset),
    }, top10_rows
