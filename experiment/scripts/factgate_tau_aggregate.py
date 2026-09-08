#!/usr/bin/env python3
"""Aggregate frozen FactGate sparsity shards. Pure NumPy; no model imports."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

MIN_BASELINE_CORRECT = 20
MIN_SENSITIVE = 10
SPARSITY_GATE = 0.20

def frac80(influences):
    x = sorted([float(v) for v in influences if float(v) > 0], reverse=True)
    if not x:
        return None
    total = sum(x)
    target = 0.8 * total
    s = 0.0
    for k,v in enumerate(x,1):
        s += v
        if s + 1e-12 >= target:
            return k
    return len(x)

def main(args):
    root = Path(args.input_root)
    rows = []
    for p in sorted(root.rglob("factgate_shard_*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            rows.extend(json.loads(x) for x in f if x.strip())
    rows.sort(key=lambda r:int(r["index"]))
    if not rows:
        raise RuntimeError("No shard outputs")
    idx = [int(r["index"]) for r in rows]
    if len(set(idx)) != len(idx):
        raise RuntimeError("Duplicate candidate indices")
    if idx != list(range(len(rows))):
        raise RuntimeError(f"Non-exact candidate coverage: {idx[:5]} ... {idx[-5:]} n={len(idx)}")

    correct = [r for r in rows if r.get("baseline_correct")]
    sensitive = []
    key_stats = defaultdict(list)
    for r in correct:
        infl = [float(f.get("influence",0)) for f in r.get("facts",[])]
        if any(v > 0 for v in infl):
            k = frac80(infl)
            frac = k / max(1, len(infl))
            r["k80"] = k
            r["fraction80"] = frac
            r["total_influence"] = float(sum(infl))
            r["mean_influence"] = float(np.mean(infl)) if infl else 0.0
            r["max_influence"] = float(max(infl)) if infl else 0.0
            sensitive.append(r)
        for f in r.get("facts",[]):
            semantic = ".".join([p for p in f.get("path",[]) if not str(p).isdigit()])
            key_stats[f"{f.get('tool_name')}.{semantic}"].append(float(f.get("influence",0)))

    n_correct = len(correct); n_sensitive = len(sensitive)
    result = {
        "n_candidates": len(rows),
        "n_baseline_correct": n_correct,
        "baseline_tool_accuracy": n_correct / len(rows),
        "n_sensitive": n_sensitive,
        "sensitive_fraction_of_baseline_correct": n_sensitive / n_correct if n_correct else 0.0,
        "min_baseline_correct_gate": MIN_BASELINE_CORRECT,
        "min_sensitive_gate": MIN_SENSITIVE,
        "sparsity_gate_fraction80_median": SPARSITY_GATE,
    }
    if sensitive:
        fracs = [r["fraction80"] for r in sensitive]
        result["median_fraction80"] = float(np.median(fracs))
        result["mean_fraction80"] = float(np.mean(fracs))
        result["median_total_influence"] = float(np.median([r["total_influence"] for r in sensitive]))
        result["median_max_fact_influence"] = float(np.median([r["max_influence"] for r in sensitive]))
        result["fraction_sensitive_passing_20pct_individually"] = float(np.mean([x <= SPARSITY_GATE + 1e-12 for x in fracs]))
    else:
        result["median_fraction80"] = None

    if n_correct < MIN_BASELINE_CORRECT:
        decision = "INCONCLUSIVE_WEAK_PROBE"
    elif n_sensitive < MIN_SENSITIVE:
        decision = "KILL_NOT_ENOUGH_SENSITIVE_DECISIONS"
    elif result["median_fraction80"] <= SPARSITY_GATE + 1e-12:
        decision = "CONTINUE"
    else:
        decision = "KILL_DIFFUSE_DEPENDENCE"
    result["decision"] = decision

    ranked = []
    for k, vals in key_stats.items():
        if len(vals) >= 2:
            ranked.append({"field":k,"n":len(vals),"mean_influence":float(np.mean(vals)),
                           "nonzero_rate":float(np.mean(np.asarray(vals)>0))})
    ranked.sort(key=lambda z:(-z["mean_influence"], -z["n"], z["field"]))
    result["top_fields"] = ranked[:15]

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (outdir/"decisions.jsonl").open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

    lines = [
        "# FactGate causal-fact sparsity pilot","",
        f"## Decision: **{decision}**","",
        "Frozen question: among consequential write decisions that the local probe reproduces at baseline, is action sensitivity concentrated in a very small subset of structured tool-return facts?","",
        "## Primary results","",
        "| Quantity | Value |","|---|---:|",
        f"| Candidate consequential decisions | {len(rows)} |",
        f"| Baseline-correct tool decisions | {n_correct} |",
        f"| Baseline tool accuracy | {result['baseline_tool_accuracy']:.3f} |",
        f"| Sensitive decisions | {n_sensitive} |",
        f"| Sensitive / baseline-correct | {result['sensitive_fraction_of_baseline_correct']:.3f} |",
    ]
    if result["median_fraction80"] is not None:
        lines += [
            f"| Median fraction of facts needed for 80% of influence | {result['median_fraction80']:.3f} |",
            f"| Mean fraction of facts needed for 80% of influence | {result['mean_fraction80']:.3f} |",
            f"| Sensitive decisions individually <=20% | {result['fraction_sensitive_passing_20pct_individually']:.3f} |",
            f"| Median total influence | {result['median_total_influence']:.3f} |",
            f"| Median max single-fact influence | {result['median_max_fact_influence']:.3f} |",
        ]
    lines += [
        "","## Frozen gate","",
        f"CONTINUE only if baseline-correct decisions >= {MIN_BASELINE_CORRECT}, sensitive decisions >= {MIN_SENSITIVE}, and median 80%-coverage fact fraction <= {SPARSITY_GATE:.2f}.",
        "","## Interpretation guardrail","",
        "This is a phenomenon pilot on historical tau-bench airline states. It tests sparsity of decision dependence under same-field factual counterfactuals. It is not a final agent-safety benchmark and does not claim that the historical trajectories themselves are current tau3-bench results.",
    ]
    if ranked:
        lines += ["","## Most influential repeated fields (descriptive)","",
                  "| Field | n | Mean influence | Nonzero rate |","|---|---:|---:|---:|"]
        for z in ranked[:10]:
            lines.append(f"| {z['field']} | {z['n']} | {z['mean_influence']:.3f} | {z['nonzero_rate']:.3f} |")
    report = "\n".join(lines)+"\n"
    (outdir/"report.md").write_text(report,encoding="utf-8")
    print(report,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",required=True)
    ap.add_argument("--outdir",required=True)
    main(ap.parse_args())
