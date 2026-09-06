#!/usr/bin/env python3
"""Aggregate execution shards for the frozen GraphGuard TempBench gate."""
import argparse, json
from pathlib import Path
import numpy as np

from experiment.scripts.graphguard_tempbench_pilot import norm, boot_diff, MIN_GOLD_CORRECT, MODEL, DATA_REVISION, BOOTSTRAPS

CONDITIONS = ["gold", "distractor", "stale", "random_object", "random_time"]


def main(args):
    root = Path(args.input_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(root.rglob("predictions_shard_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            rows.extend(json.loads(x) for x in f if x.strip())
    rows.sort(key=lambda r: int(r["index"]))
    if not rows:
        raise RuntimeError("No shard predictions found")
    indices = [int(r["index"]) for r in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError(f"Shard coverage is not exact: n={len(rows)} first={indices[:5]} last={indices[-5:]}")

    gold_correct = [r for r in rows if r["correct_gold"]]
    n, ng = len(rows), len(gold_correct)
    result = {
        "model": MODEL, "dataset_revision": DATA_REVISION, "n_evaluated": n,
        "n_gold_correct": ng, "gold_accuracy": ng / n,
        "minimum_gold_correct_for_interpretation": MIN_GOLD_CORRECT,
        "execution": "5 deterministic inference shards; exact frozen sample and controls",
    }
    for c in CONDITIONS[1:]:
        if ng:
            flips = [norm(r[f"pred_{c}"]) != norm(r["pred_gold"]) for r in gold_correct]
            result[f"flip_rate_{c}"] = float(np.mean(flips))
            result[f"accuracy_{c}"] = float(np.mean([r[f"correct_{c}"] for r in rows]))

    if ng >= MIN_GOLD_CORRECT:
        dist = np.array([norm(r["pred_distractor"]) != norm(r["pred_gold"]) for r in gold_correct])
        ro = np.array([norm(r["pred_random_object"]) != norm(r["pred_gold"]) for r in gold_correct])
        stale = np.array([norm(r["pred_stale"]) != norm(r["pred_gold"]) for r in gold_correct])
        rt = np.array([norm(r["pred_random_time"]) != norm(r["pred_gold"]) for r in gold_correct])
        dd, dci = boot_diff(dist, ro)
        sd, sci = boot_diff(stale, rt)
        result["distractor_minus_random_object"] = {"delta": dd, "ci95": dci}
        result["stale_minus_random_time"] = {"delta": sd, "ci95": sci}
        result["pass_distractor"] = bool(dist.mean() >= .10 and dci[0] > 0)
        result["pass_stale"] = bool(stale.mean() >= .10 and sci[0] > 0)
        result["decision"] = "CONTINUE" if (result["pass_distractor"] or result["pass_stale"]) else "KILL"
        stable = [r["correct_gold"] and norm(r["pred_distractor"]) == norm(r["pred_gold"]) and norm(r["pred_stale"]) == norm(r["pred_gold"]) for r in rows]
        result["stable_accuracy_both_plausible"] = float(np.mean(stable))
    else:
        result["decision"] = "INCONCLUSIVE_WEAK_MODEL"

    (outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(outdir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = [
        "# GraphGuard TempBench phenomenon pilot",
        "", f"## Decision: **{result['decision']}**", "",
        "Frozen gate: among questions answered correctly with gold evidence, a plausible single-edge intervention must flip at least 10% of decisions and exceed its matched random control with a paired-bootstrap 95% CI entirely above zero.",
        "", "## Primary results", "", "| Quantity | Value |", "|---|---:|",
        f"| Eligible evaluated | {n} |", f"| Gold-correct | {ng} |", f"| Gold accuracy | {result['gold_accuracy']:.4f} |",
    ]
    for c in CONDITIONS[1:]:
        if f"flip_rate_{c}" in result:
            lines.append(f"| Flip rate — {c} | {result[f'flip_rate_{c}']:.4f} |")
    if "distractor_minus_random_object" in result:
        z = result["distractor_minus_random_object"]
        lines.append(f"| Distractor − random-object flip rate | {z['delta']:.4f} [{z['ci95'][0]:.4f}, {z['ci95'][1]:.4f}] |")
        z = result["stale_minus_random_time"]
        lines.append(f"| Stale − random-time flip rate | {z['delta']:.4f} [{z['ci95'][0]:.4f}, {z['ci95'][1]:.4f}] |")
        lines.append(f"| Stable accuracy under both plausible interventions | {result['stable_accuracy_both_plausible']:.4f} |")
    lines += ["", "## Execution", "", "The exact frozen cohort was evaluated in five deterministic parallel shards. Sharding changed compute scheduling only; sample selection, controls, prompts, model, decoding, and statistical gate were unchanged."]
    report = "\n".join(lines) + "\n"
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--outdir", required=True)
    main(ap.parse_args())
