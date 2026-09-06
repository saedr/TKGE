#!/usr/bin/env python3
"""Execution-only sharding for the frozen GraphGuard TempBench pilot."""
import argparse, json
from collections import defaultdict
from pathlib import Path
from huggingface_hub import hf_hub_download

from experiment.scripts.graphguard_tempbench_pilot import (
    DATASET, DATA_REVISION, jloadl, prepare_items, attach_controls,
    load_model, infer, make_prompt, is_correct,
)

CONDITIONS = ["gold", "distractor", "stale", "random_object", "random_time"]


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bench_path = hf_hub_download(DATASET, "benchmark/benchmark_labelled.jsonl", repo_type="dataset", revision=DATA_REVISION)
    flags_path = hf_hub_download(DATASET, "benchmark/functional_negatives.jsonl", repo_type="dataset", revision=DATA_REVISION)
    benchmark, flags = jloadl(bench_path), jloadl(flags_path)
    items, selection, answer_key = prepare_items(benchmark, flags)
    attach_controls(items)  # before sharding: preserves exact frozen random controls
    chosen = [(i, it) for i, it in enumerate(items) if i % args.num_shards == args.shard_index]
    print(f"SHARD {args.shard_index}/{args.num_shards}: total_selected={len(items)} shard_items={len(chosen)} selection={selection}", flush=True)

    prompts, order = [], []
    for i, it in chosen:
        for c in CONDITIONS:
            prompts.append(make_prompt(it["q"], it["conditions"][c]))
            order.append((i, c))

    tok, model = load_model()
    preds = infer(tok, model, prompts)
    predmap = defaultdict(dict)
    for (i, c), p in zip(order, preds):
        predmap[i][c] = p

    rows = []
    for i, it in chosen:
        row = {
            "index": i,
            "id": it["q"].get("id"),
            "question": it["q"].get("question"),
            "golds": it["golds"],
        }
        for c in CONDITIONS:
            row[f"pred_{c}"] = predmap[i][c]
            row[f"correct_{c}"] = is_correct(predmap[i][c], it["golds"])
        rows.append(row)

    with open(outdir / f"predictions_shard_{args.shard_index}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (outdir / f"selection_shard_{args.shard_index}.json").write_text(
        json.dumps({"selection": selection, "answer_key": answer_key, "total_selected": len(items), "shard_items": len(chosen)}, indent=2), encoding="utf-8"
    )
    print(f"SHARD COMPLETE {args.shard_index}: rows={len(rows)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    main(ap.parse_args())
