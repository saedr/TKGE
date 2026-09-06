#!/usr/bin/env python3
"""Frozen phenomenon gate for GraphGuard on TempBench.

No method training. A single fixed Qwen3-1.7B generator answers the same test
questions under gold evidence, a functional stale single-edge replacement, a
functional relation-matched distractor replacement, and two matched random
controls. The scientific gate is frozen before inference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = "Guen/tempbench"
DATA_REVISION = "ad8ea76"
MODEL = "Qwen/Qwen3-1.7B"
MAX_ITEMS = 1000
SEED = 42
BOOTSTRAPS = 5000
MIN_GOLD_CORRECT = 100


def jloadl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def canon_edge(e):
    return (
        str(e.get("s", "")), str(e.get("r", "")), str(e.get("o", "")),
        str(e.get("t_start", "")), str(e.get("t_end", "")),
    )


def one_edge_replacement(gold, alt):
    """Return (gold_edge, alt_edge) iff alt replaces exactly one gold edge."""
    gs = {canon_edge(x): x for x in gold}
    aa = {canon_edge(x): x for x in alt}
    removed = list(set(gs) - set(aa))
    added = list(set(aa) - set(gs))
    if len(removed) != 1 or len(added) != 1 or len(gold) != len(alt):
        return None
    return copy.deepcopy(gs[removed[0]]), copy.deepcopy(aa[added[0]])


def is_interval(q):
    for k, v in q.items():
        lk = str(k).lower()
        if any(tok in lk for tok in ("operator", "question_type", "template", "temporal_type", "type")):
            if isinstance(v, str) and "interval" in v.lower():
                return True
    return False


def answer_values(q):
    for k in ("answer", "answers", "gold_answer", "gold_answers", "answer_label", "target_answer"):
        if k not in q:
            continue
        v = q[k]
        vals = v if isinstance(v, list) else [v]
        out = []
        for z in vals:
            if isinstance(z, dict):
                for kk in ("label", "answer", "text", "name", "value"):
                    if kk in z:
                        out.append(str(z[kk]))
                        break
            elif z is not None:
                out.append(str(z))
        if out:
            return out, k
    raise KeyError(f"No answer field found. keys={sorted(q.keys())}")


def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.S)
    s = s.replace("answer:", " ").replace("final answer:", " ")
    s = s.splitlines()[0] if s.splitlines() else s
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(pred, golds):
    p = norm(pred)
    return any(p == norm(g) for g in golds)


def replace_edge(gold, target, replacement):
    target_key = canon_edge(target)
    out = []
    done = False
    for e in gold:
        if not done and canon_edge(e) == target_key:
            out.append(copy.deepcopy(replacement))
            done = True
        else:
            out.append(copy.deepcopy(e))
    if not done:
        raise RuntimeError("target edge not found")
    return out


def choose_random_object(donor_edges, target, forbidden):
    for e in donor_edges:
        obj = str(e.get("o", ""))
        if obj and obj not in forbidden:
            x = copy.deepcopy(target)
            x["o"] = obj
            return x
    raise RuntimeError("could not build random object control")


def choose_random_time(donor_edges, target, forbidden_times):
    for e in donor_edges:
        ts = (str(e.get("t_start", "")), str(e.get("t_end", "")))
        if ts not in forbidden_times and any(ts):
            x = copy.deepcopy(target)
            x["t_start"] = e.get("t_start")
            x["t_end"] = e.get("t_end")
            return x
    # deterministic fallback: extreme synthetic shift, used only if donor times collide
    x = copy.deepcopy(target)
    try:
        y = int(float(target.get("t_start")))
        x["t_start"] = y + 97
        x["t_end"] = y + 97
    except Exception:
        x["t_start"] = "9999"
        x["t_end"] = "9999"
    return x


def evidence_text(edges):
    rows = []
    for i, e in enumerate(edges, 1):
        t0, t1 = e.get("t_start"), e.get("t_end")
        when = str(t0) if t0 == t1 else f"{t0} to {t1}"
        rows.append(f"{i}. {e.get('s')} -- {e.get('r')} --> {e.get('o')} [time: {when}]")
    return "\n".join(rows)


def make_prompt(q, edges):
    return (
        "Answer the temporal knowledge-graph question using only the evidence. "
        "Do not use outside knowledge. Return only the final entity name or year, with no explanation.\n\n"
        f"Question: {q['question']}\n\nEvidence:\n{evidence_text(edges)}\n\nAnswer:"
    )


def prepare_items(benchmark, flags):
    flagmap = {str(x.get("id")): x for x in flags}
    candidates = []
    diagnostics = Counter()
    answer_key = None
    for q in benchmark:
        if q.get("split") != "test":
            continue
        diagnostics["test"] += 1
        if is_interval(q):
            diagnostics["interval_excluded"] += 1
            continue
        f = flagmap.get(str(q.get("id")), {})
        if not (f.get("dist_functional") and f.get("stale_functional")):
            diagnostics["nonfunctional_excluded"] += 1
            continue
        gold = q.get("S_star") or []
        dist = q.get("S_dist") or []
        stale = q.get("S_stale") or []
        dr = one_edge_replacement(gold, dist)
        sr = one_edge_replacement(gold, stale)
        if not dr or not sr:
            diagnostics["not_single_edge_excluded"] += 1
            continue
        dtarget, dedge = dr
        starget, sedge = sr
        if canon_edge(dtarget) != canon_edge(starget):
            diagnostics["different_target_excluded"] += 1
            continue
        golds, answer_key = answer_values(q)
        candidates.append({
            "q": q, "golds": golds, "target": dtarget,
            "dist_edge": dedge, "stale_edge": sedge,
        })
    # deterministic pseudo-random subset independent of model outcomes
    candidates.sort(key=lambda z: hashlib.sha256((str(z["q"].get("id")) + "|42").encode()).hexdigest())
    if len(candidates) > MAX_ITEMS:
        candidates = candidates[:MAX_ITEMS]
    diagnostics["eligible_selected"] = len(candidates)
    return candidates, dict(diagnostics), answer_key


def attach_controls(items):
    rng = random.Random(SEED)
    all_edges = []
    for it in items:
        all_edges.extend(it["q"]["S_star"])
    for idx, it in enumerate(items):
        donors = all_edges.copy()
        rng.shuffle(donors)
        target = it["target"]
        forbidden_obj = {str(target.get("o", "")), str(it["dist_edge"].get("o", ""))}
        rand_obj_edge = choose_random_object(donors, target, forbidden_obj)
        forbidden_times = {
            (str(target.get("t_start", "")), str(target.get("t_end", ""))),
            (str(it["stale_edge"].get("t_start", "")), str(it["stale_edge"].get("t_end", ""))),
        }
        rand_time_edge = choose_random_time(donors, target, forbidden_times)
        gold = it["q"]["S_star"]
        it["conditions"] = {
            "gold": gold,
            "distractor": replace_edge(gold, target, it["dist_edge"]),
            "stale": replace_edge(gold, target, it["stale_edge"]),
            "random_object": replace_edge(gold, target, rand_obj_edge),
            "random_time": replace_edge(gold, target, rand_time_edge),
        }
    return items


def load_model():
    print(f"Loading {MODEL} on CPU", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    return tok, model


def infer(tok, model, prompts, batch_size=8):
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start+batch_size]
        rendered = []
        for p in batch_prompts:
            msgs = [{"role": "user", "content": p}]
            try:
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                s += " /no_think"
            rendered.append(s)
        enc = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=768)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        for i in range(len(batch_prompts)):
            new = gen[i, enc["input_ids"].shape[1]:]
            text = tok.decode(new, skip_special_tokens=True).strip()
            # Qwen occasionally leaves a think block despite no-think; keep only post-think text.
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            outputs.append(text)
        done = min(start + batch_size, len(prompts))
        if done % 80 == 0 or done == len(prompts):
            print(f"INFER {done}/{len(prompts)}", flush=True)
    return outputs


def boot_diff(a, b, n_boot=BOOTSTRAPS):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(SEED)
    n = len(a)
    diffs = np.empty(n_boot)
    for j in range(n_boot):
        ix = rng.integers(0, n, n)
        diffs[j] = a[ix].mean() - b[ix].mean()
    return float(a.mean() - b.mean()), [float(x) for x in np.quantile(diffs, [0.025, 0.975])]


def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    bench_path = hf_hub_download(DATASET, "benchmark/benchmark_labelled.jsonl", repo_type="dataset", revision=DATA_REVISION)
    flags_path = hf_hub_download(DATASET, "benchmark/functional_negatives.jsonl", repo_type="dataset", revision=DATA_REVISION)
    benchmark, flags = jloadl(bench_path), jloadl(flags_path)
    print(f"FIRST_KEYS {sorted(benchmark[0].keys())}", flush=True)
    items, selection, answer_key = prepare_items(benchmark, flags)
    print(f"SELECTION {selection} answer_key={answer_key}", flush=True)
    if not items:
        raise RuntimeError("No eligible items after frozen filters")
    attach_controls(items)

    conditions = ["gold", "distractor", "stale", "random_object", "random_time"]
    prompts = []
    order = []
    # Interleave by item then condition to prevent any temporal batching artifact by condition.
    for i, it in enumerate(items):
        for c in conditions:
            prompts.append(make_prompt(it["q"], it["conditions"][c]))
            order.append((i, c))

    tok, model = load_model()
    preds = infer(tok, model, prompts)
    predmap = defaultdict(dict)
    for (i, c), p in zip(order, preds):
        predmap[i][c] = p

    rows = []
    for i, it in enumerate(items):
        row = {
            "id": it["q"].get("id"),
            "question": it["q"].get("question"),
            "golds": it["golds"],
        }
        for c in conditions:
            row[f"pred_{c}"] = predmap[i][c]
            row[f"correct_{c}"] = is_correct(predmap[i][c], it["golds"])
        rows.append(row)

    gold_correct = [r for r in rows if r["correct_gold"]]
    n = len(rows); ng = len(gold_correct)
    gold_acc = ng / n
    result = {
        "model": MODEL, "dataset_revision": DATA_REVISION, "selection": selection,
        "n_evaluated": n, "n_gold_correct": ng, "gold_accuracy": gold_acc,
        "minimum_gold_correct_for_interpretation": MIN_GOLD_CORRECT,
    }

    for c in conditions[1:]:
        if ng:
            flips = [norm(r[f"pred_{c}"]) != norm(r["pred_gold"]) for r in gold_correct]
            result[f"flip_rate_{c}"] = float(np.mean(flips))
            result[f"accuracy_{c}"] = float(np.mean([r[f"correct_{c}"] for r in rows]))

    if ng >= MIN_GOLD_CORRECT:
        dist_flip = np.array([norm(r["pred_distractor"]) != norm(r["pred_gold"]) for r in gold_correct])
        rand_obj = np.array([norm(r["pred_random_object"]) != norm(r["pred_gold"]) for r in gold_correct])
        stale_flip = np.array([norm(r["pred_stale"]) != norm(r["pred_gold"]) for r in gold_correct])
        rand_time = np.array([norm(r["pred_random_time"]) != norm(r["pred_gold"]) for r in gold_correct])
        ddelta, dci = boot_diff(dist_flip, rand_obj)
        sdelta, sci = boot_diff(stale_flip, rand_time)
        result["distractor_minus_random_object"] = {"delta": ddelta, "ci95": dci}
        result["stale_minus_random_time"] = {"delta": sdelta, "ci95": sci}
        pass_dist = dist_flip.mean() >= 0.10 and dci[0] > 0
        pass_stale = stale_flip.mean() >= 0.10 and sci[0] > 0
        result["pass_distractor"] = bool(pass_dist)
        result["pass_stale"] = bool(pass_stale)
        result["decision"] = "CONTINUE" if (pass_dist or pass_stale) else "KILL"
        stable_both = [r["correct_gold"] and norm(r["pred_distractor"]) == norm(r["pred_gold"]) and norm(r["pred_stale"]) == norm(r["pred_gold"]) for r in rows]
        result["stable_accuracy_both_plausible"] = float(np.mean(stable_both))
    else:
        result["decision"] = "INCONCLUSIVE_WEAK_MODEL"

    with open(outdir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# GraphGuard TempBench phenomenon pilot",
        "",
        f"## Decision: **{result['decision']}**",
        "",
        "Frozen gate: among questions answered correctly with gold evidence, a plausible single-edge intervention must flip at least 10% of decisions and exceed its matched random control with a paired-bootstrap 95% CI entirely above zero.",
        "",
        "## Primary results",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Eligible evaluated | {n} |",
        f"| Gold-correct | {ng} |",
        f"| Gold accuracy | {gold_acc:.4f} |",
    ]
    for c in conditions[1:]:
        if f"flip_rate_{c}" in result:
            lines.append(f"| Flip rate — {c} | {result[f'flip_rate_{c}']:.4f} |")
    if "distractor_minus_random_object" in result:
        z = result["distractor_minus_random_object"]
        lines.append(f"| Distractor − random-object flip rate | {z['delta']:.4f} [{z['ci95'][0]:.4f}, {z['ci95'][1]:.4f}] |")
        z = result["stale_minus_random_time"]
        lines.append(f"| Stale − random-time flip rate | {z['delta']:.4f} [{z['ci95'][0]:.4f}, {z['ci95'][1]:.4f}] |")
        lines.append(f"| Stable accuracy under both plausible interventions | {result['stable_accuracy_both_plausible']:.4f} |")
    lines += [
        "", "## Frozen setup", "",
        f"- Model: `{MODEL}`; greedy decoding; no fine-tuning.",
        "- TempBench test split only; interval questions excluded.",
        "- Both typed negatives must be functional, single-edge replacements, and target the same gold edge.",
        f"- At most {MAX_ITEMS} eligible items, selected deterministically before inference.",
        "- Distractor control: same target edge with a random object drawn from another gold edge.",
        "- Stale control: same target fact with a random time drawn from another gold edge.",
        f"- {BOOTSTRAPS} paired bootstrap replicates over gold-correct items.",
        "- If KILL: do not build GraphGuard from this phenomenon without a new independent rationale/data source.",
        "", "## Selection diagnostics", "", "```json", json.dumps(selection, indent=2), "```",
    ]
    report = "\n".join(lines) + "\n"
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    main(ap.parse_args())
