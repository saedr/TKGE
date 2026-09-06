#!/usr/bin/env python3
"""Feasibility gate for prospective KGE-reliability study on EMERGE.

No KGE is trained here. We construct natural Exists-vs-Deprecate cohorts for
2024 and 2025, verify that targets existed in the corresponding Jan-1 KG
snapshot, compute only pre-outcome graph/history controls, and run a simple
2024->2025 logistic baseline.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = "klimzaporojets/emerge-benchmark"
DELTA_SUFFIXES = ["01-08", "01-15", "01-22", "01-29", "02-05"]
YEARS = [2023, 2024, 2025]
TARGET_YEARS = [2024, 2025]


def download_inputs(workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    paths = {"corpus": defaultdict(list), "kg": {}}
    for year in YEARS:
        for suffix in DELTA_SUFFIXES:
            fn = f"corpus/snapshot_{year}-01-01/delta_{year}-{suffix}.jsonl"
            p = hf_hub_download(REPO, fn, repo_type="dataset", local_dir=str(workdir))
            paths["corpus"][year].append(Path(p))
    for year in TARGET_YEARS:
        fn = f"kg_snapshots/kg_snapshot_{year}-01-01.tsv.gz"
        p = hf_hub_download(REPO, fn, repo_type="dataset", local_dir=str(workdir))
        paths["kg"][year] = Path(p)
    return paths


def assessment_map(entry):
    vals = entry.get("llm_assessment")
    if vals is None:
        vals = entry.get("assessments", [])
    out = {}
    for item in vals or []:
        name = item.get("llm_name")
        if name:
            out[name] = bool(item.get("llm_assessment"))
    return out


def preferred_supported(entry):
    """Use the strongest available assessment without discarding uncovered records."""
    m = assessment_map(entry)
    for name in (
        "Meta-Llama-3.1-405B_prompt_v1",
        "Meta-Llama-3.1-405B",
        "Meta-Llama-3.1-8B",
    ):
        if name in m:
            return m[name], name
    return False, "none"


def eightb_supported(entry):
    m = assessment_map(entry)
    return bool(m.get("Meta-Llama-3.1-8B", False))


def added_date(entry):
    span = entry.get("triple_lifespan_date")
    if isinstance(span, list) and span:
        return span[0]
    return entry.get("added_date")


def parse_iso_day(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None


def collect_year(year: int, files):
    """Collapse repeated passage/delta observations to one row per KG triple."""
    state = {}
    stats = Counter()
    assessor = Counter()
    eightb_ops = Counter()
    raw_ops = Counter()
    preferred_ops = Counter()
    records = 0
    entries = 0

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records += 1
                obj = json.loads(line)
                for entry in obj.get("tkgu_triples", []):
                    entries += 1
                    triple = entry.get("triple")
                    if not isinstance(triple, list) or len(triple) != 3 or not all(isinstance(x, str) for x in triple):
                        stats["malformed_triple"] += 1
                        continue
                    ops = entry.get("tkgu_operations") or []
                    if isinstance(ops, str):
                        ops = [ops]
                    for op in ops:
                        raw_ops[op] += 1
                    if eightb_supported(entry):
                        for op in ops:
                            eightb_ops[op] += 1
                    ok, source = preferred_supported(entry)
                    assessor[source] += 1
                    if not ok:
                        continue
                    for op in ops:
                        preferred_ops[op] += 1
                    has_exists = "x-triples" in ops
                    has_dep = "d-triples" in ops
                    if not has_exists and not has_dep:
                        continue
                    key = tuple(triple)
                    row = state.setdefault(key, {
                        "triple": key,
                        "exists": False,
                        "deprecate": False,
                        "added_dates": [],
                        "observations": 0,
                    })
                    row["exists"] = row["exists"] or has_exists
                    row["deprecate"] = row["deprecate"] or has_dep
                    row["observations"] += 1
                    ad = parse_iso_day(added_date(entry))
                    if ad is not None:
                        row["added_dates"].append(ad)

    cohort = []
    ambiguous = 0
    for key, row in state.items():
        if row["exists"] and row["deprecate"]:
            ambiguous += 1
            continue
        label = 1 if row["deprecate"] else 0
        ad = min(row["added_dates"]) if row["added_dates"] else None
        cohort.append({
            "year": year,
            "h": key[0],
            "r": key[1],
            "t": key[2],
            "label": label,
            "added_date": ad.isoformat() if ad else None,
            "observations": row["observations"],
        })

    return cohort, {
        "records": records,
        "tkgu_entries": entries,
        "raw_operations": dict(raw_ops),
        "eightb_supported_operations": dict(eightb_ops),
        "preferred_supported_operations": dict(preferred_ops),
        "preferred_assessor_usage": dict(assessor),
        "unique_candidate_triples": len(state),
        "ambiguous_exists_and_deprecate": ambiguous,
        "cohort_unique_triples": len(cohort),
    }


def historical_relation_rates(cohorts, years):
    counts = defaultdict(lambda: [0, 0])
    for year in years:
        for row in cohorts[year]:
            counts[row["r"]][int(row["label"])] += 1
    rates = {}
    for rel, (neg, pos) in counts.items():
        rates[rel] = (pos + 1.0) / (neg + pos + 2.0)
    return rates, {rel: {"exists": v[0], "deprecate": v[1]} for rel, v in counts.items()}


def scan_snapshot(path: Path, cohort):
    target_triples = {(r["h"].encode(), r["r"].encode(), r["t"].encode()) for r in cohort}
    needed_entities = {x for tri in target_triples for x in (tri[0], tri[2])}
    needed_relations = {tri[1] for tri in target_triples}
    degrees = Counter()
    relfreq = Counter()
    found = set()
    n_lines = 0
    malformed = 0
    with gzip.open(path, "rb") as f:
        for line in f:
            n_lines += 1
            parts = line.rstrip(b"\r\n").split(b"\t")
            if len(parts) < 3:
                malformed += 1
                continue
            h, r, t = parts[0], parts[1], parts[2]
            if r in needed_relations:
                relfreq[r] += 1
            if h in needed_entities:
                degrees[h] += 1
            if t in needed_entities:
                degrees[t] += 1
            tri = (h, r, t)
            if tri in target_triples:
                found.add(tri)
    return {
        "n_snapshot_triples": n_lines - malformed,
        "malformed_lines": malformed,
        "found": found,
        "degrees": degrees,
        "relfreq": relfreq,
    }


def enrich(year, cohort, scan, hist_rates):
    snapshot_day = date(year, 1, 1)
    out = []
    future_age_violations = 0
    for row in cohort:
        hb, rb, tb = row["h"].encode(), row["r"].encode(), row["t"].encode()
        ad = parse_iso_day(row.get("added_date"))
        if ad is None:
            age = np.nan
        else:
            age = (snapshot_day - ad).days
            if age < 0:
                future_age_violations += 1
                age = np.nan
        found = (hb, rb, tb) in scan["found"]
        out.append({
            **row,
            "in_snapshot": found,
            "head_degree": int(scan["degrees"].get(hb, 0)),
            "tail_degree": int(scan["degrees"].get(tb, 0)),
            "relation_frequency": int(scan["relfreq"].get(rb, 0)),
            "age_days": None if np.isnan(age) else int(age),
            "history_deprecation_rate": float(hist_rates.get(row["r"], 0.5)),
        })
    return out, future_age_violations


def cohort_summary(rows):
    n = len(rows)
    pos = sum(r["label"] for r in rows)
    neg = n - pos
    pos_rels = Counter(r["r"] for r in rows if r["label"] == 1)
    all_rels = Counter(r["r"] for r in rows)
    top_pos = pos_rels.most_common(10)
    membership = {
        str(label): {
            "n": sum(1 for r in rows if r["label"] == label),
            "found": sum(1 for r in rows if r["label"] == label and r["in_snapshot"]),
        }
        for label in (0, 1)
    }
    for v in membership.values():
        v["rate"] = v["found"] / v["n"] if v["n"] else None
    return {
        "n": n,
        "exists": neg,
        "deprecate": pos,
        "prevalence": pos / n if n else None,
        "relations_total": len(all_rels),
        "relations_with_deprecate": len(pos_rels),
        "top_positive_relations": top_pos,
        "top1_positive_share": (top_pos[0][1] / pos) if pos and top_pos else None,
        "snapshot_membership": membership,
    }


def rows_to_df(rows):
    df = pd.DataFrame([r for r in rows if r["in_snapshot"]]).copy()
    for col in ("head_degree", "tail_degree", "relation_frequency"):
        df[f"log_{col}"] = np.log1p(df[col].astype(float))
    df["log_age_days"] = np.log1p(pd.to_numeric(df["age_days"], errors="coerce").clip(lower=0))
    df["age_missing"] = df["age_days"].isna().astype(int)
    return df


def fit_baselines(train_rows, test_rows):
    train = rows_to_df(train_rows)
    test = rows_to_df(test_rows)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise RuntimeError("Both labels are required in 2024 train and 2025 test cohorts")

    numeric = ["log_head_degree", "log_tail_degree", "log_relation_frequency", "log_age_days", "age_missing", "history_deprecation_rate"]
    variants = {
        "structural_history": (numeric, []),
        "relation_only": ([], ["r"]),
        "full_controls": (numeric, ["r"]),
    }
    results = {}
    for name, (nums, cats) in variants.items():
        transformers = []
        if nums:
            transformers.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=False)),
                ("scale", StandardScaler()),
            ]), nums))
        if cats:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
        prep = ColumnTransformer(transformers)
        model = Pipeline([
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
        ])
        model.fit(train, y_train)
        p = model.predict_proba(test)[:, 1]
        results[name] = {
            "auprc": float(average_precision_score(y_test, p)),
            "auroc": float(roc_auc_score(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
        }
    results["prevalence_baseline"] = {
        "auprc": float(y_test.mean()),
        "prevalence": float(y_test.mean()),
    }
    return results


def write_csv_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(args):
    workdir = Path(args.workdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = download_inputs(workdir)

    cohorts = {}
    source_stats = {}
    for year in YEARS:
        cohorts[year], source_stats[year] = collect_year(year, paths["corpus"][year])
        print(f"YEAR {year}: cohort={len(cohorts[year])} stats={source_stats[year]}", flush=True)

    hist_2024, hist_counts_2024 = historical_relation_rates(cohorts, [2023])
    hist_2025, hist_counts_2025 = historical_relation_rates(cohorts, [2023, 2024])

    enriched = {}
    scan_stats = {}
    age_violations = {}
    for year, hist in ((2024, hist_2024), (2025, hist_2025)):
        print(f"Scanning KG snapshot {year}...", flush=True)
        scan = scan_snapshot(paths["kg"][year], cohorts[year])
        enriched[year], age_violations[year] = enrich(year, cohorts[year], scan, hist)
        scan_stats[year] = {
            "n_snapshot_triples": scan["n_snapshot_triples"],
            "malformed_lines": scan["malformed_lines"],
            "target_triples_found": len(scan["found"]),
        }
        print(f"SNAPSHOT {year}: {scan_stats[year]}", flush=True)

    summaries = {year: cohort_summary(enriched[year]) for year in TARGET_YEARS}
    baselines = fit_baselines(enriched[2024], enriched[2025])

    train_keys = {(r["h"], r["r"], r["t"]) for r in enriched[2024] if r["in_snapshot"]}
    test_keys = {(r["h"], r["r"], r["t"]) for r in enriched[2025] if r["in_snapshot"]}
    overlap = train_keys & test_keys
    transition_counts = Counter()
    y24 = {(r["h"], r["r"], r["t"]): r["label"] for r in enriched[2024] if r["in_snapshot"]}
    y25 = {(r["h"], r["r"], r["t"]): r["label"] for r in enriched[2025] if r["in_snapshot"]}
    for key in overlap:
        transition_counts[f"{y24[key]}->{y25[key]}"] += 1

    gate_checks = {}
    for year in TARGET_YEARS:
        s = summaries[year]
        gate_checks[f"{year}_at_least_100_deprecates"] = s["deprecate"] >= 100
        gate_checks[f"{year}_at_least_10_positive_relations"] = s["relations_with_deprecate"] >= 10
        gate_checks[f"{year}_top1_positive_share_below_60pct"] = (s["top1_positive_share"] or 1.0) < 0.60
        gate_checks[f"{year}_deprecate_snapshot_membership_at_least_95pct"] = (s["snapshot_membership"]["1"]["rate"] or 0.0) >= 0.95
        gate_checks[f"{year}_exists_snapshot_membership_at_least_95pct"] = (s["snapshot_membership"]["0"]["rate"] or 0.0) >= 0.95
    gate_checks["baseline_executes_out_of_time"] = all(math.isfinite(v["auprc"]) for k, v in baselines.items() if "auprc" in v)
    decision = "GO" if all(gate_checks.values()) else "NO-GO"

    summary = {
        "decision": decision,
        "scope": "data-feasibility only; no KGE trained",
        "forecast_horizon": "EMERGE's five progressively larger post-snapshot windows, ending about 35 days after Jan 1; this is not a one-year survival label",
        "label_definition": "Deprecate=1; Exists=0; exact triples collapsed within snapshot year; triples observed as both are excluded",
        "assessment_policy": "Meta-Llama-3.1-405B_prompt_v1 if present, else Meta-Llama-3.1-405B if present, else Meta-Llama-3.1-8B",
        "source_stats": source_stats,
        "cohort_summary": summaries,
        "snapshot_scan": scan_stats,
        "future_age_violations": age_violations,
        "historical_relation_counts": {"for_2024_from_2023": hist_counts_2024, "for_2025_from_2023_2024": hist_counts_2025},
        "baseline_2024_to_2025": baselines,
        "exact_triple_overlap_2024_2025": {"n": len(overlap), "label_transitions": dict(transition_counts)},
        "gate_checks": gate_checks,
        "feature_leakage_audit": {
            "used": ["head_degree_at_snapshot", "tail_degree_at_snapshot", "relation_frequency_at_snapshot", "triple_age_at_snapshot", "past_relation_deprecation_rate", "relation_identity"],
            "not_used": ["passage", "revision_date", "removed_date", "deprecation_reason", "qualifier_info", "future_delta_date", "future_text"],
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv_rows(outdir / "cohort_2024.csv", enriched[2024])
    write_csv_rows(outdir / "cohort_2025.csv", enriched[2025])

    lines = [
        "# Prospective revision feasibility gate",
        "",
        f"## Decision: **{decision}**",
        "",
        "This is a data-feasibility result only. No KGE was trained.",
        "",
        "Important horizon qualification: EMERGE observes five progressively larger windows after each January 1 snapshot, ending roughly 35 days later. Therefore the target is near-term invalidation, not a literal 2024-to-2025 one-year survival outcome.",
        "",
        "## Cohort health",
        "",
        "| Year | Exists | Deprecate | Prevalence | Positive relations | Top-1 positive relation share | Exists in snapshot | Deprecate in snapshot |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year in TARGET_YEARS:
        s = summaries[year]
        lines.append(
            f"| {year} | {s['exists']} | {s['deprecate']} | {s['prevalence']:.4f} | {s['relations_with_deprecate']} | "
            f"{s['top1_positive_share']:.3f} | {s['snapshot_membership']['0']['rate']:.3f} | {s['snapshot_membership']['1']['rate']:.3f} |"
        )
    lines += [
        "",
        "## Controls-only 2024 -> 2025 baseline",
        "",
        "| Baseline | 2025 AUPRC | AUROC | Brier |",
        "|---|---:|---:|---:|",
    ]
    for name in ("structural_history", "relation_only", "full_controls"):
        m = baselines[name]
        lines.append(f"| {name} | {m['auprc']:.6f} | {m['auroc']:.6f} | {m['brier']:.6f} |")
    lines.append(f"| prevalence | {baselines['prevalence_baseline']['auprc']:.6f} | — | — |")
    lines += [
        "",
        "The controls-only model is not the scientific test. Its role is to establish the baseline that a future KGE score must beat out of time.",
        "",
        "## Gate checks",
        "",
    ]
    for key, value in gate_checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — {key}")
    lines += [
        "",
        "## Leakage audit",
        "",
        "Features use only the KG snapshot and outcomes from earlier snapshot years. Future passages, removal dates, deprecation reasons, qualifiers, and future text are not features.",
        "",
        "## Next step if GO",
        "",
        "Train exactly one KGE on the 2024 snapshot, relation-normalize its score, add that score to the frozen controls baseline, and evaluate the incremental 2025 AUPRC. Do not add methods or features before that gate.",
    ]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--outdir", required=True)
    main(ap.parse_args())
