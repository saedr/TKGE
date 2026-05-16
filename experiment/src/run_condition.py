import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import load_dataset
from .evaluate import evaluate_tail_only
from .model import ComplEx, DistMult, TransE
from .perturb import perturb_training
from .stability import NOTE
from .train import train_kge
from .utils import StageTimer, ensure_dir, set_seed, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiment/configs/smoke.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--results-root", default="experiment/results")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.budget is not None:
        cfg["perturbation"]["budget"] = args.budget
    if args.model is not None:
        cfg["model"]["name"] = args.model

    condition = args.condition
    if args.tag:
        outdir = Path(args.results_root) / args.tag / condition
    else:
        outdir = Path(args.results_root) / "raw" / condition
    ensure_dir(outdir)

    timer = StageTimer()
    set_seed(cfg["seed"])
    device = torch.device(cfg["runtime"]["device"])

    with timer.timeit("data_loading"):
        ds = load_dataset(cfg, str(Path(args.results_root) / "mapping_metadata.json"))

    with timer.timeit("perturbation"):
        train_cond, perturb_meta = perturb_training(
            ds["train"],
            condition,
            cfg["perturbation"]["budget"],
            ds["orig_degree"],
            cfg["perturbation"]["smoothing_c"],
            cfg["seed"],
            valid_triples=ds["valid"],
            test_triples=ds["test"],
        )

    model_name = cfg["model"]["name"]
    if model_name == "DistMult":
        model = DistMult(ds["num_entities"], ds["num_relations"], cfg["model"]["embedding_dim"])
    elif model_name == "TransE":
        model = TransE(ds["num_entities"], ds["num_relations"], cfg["model"]["embedding_dim"])
    elif model_name == "ComplEx":
        model = ComplEx(ds["num_entities"], ds["num_relations"], cfg["model"]["embedding_dim"])
    else:
        raise ValueError(f"Unsupported model in this runner: {model_name}")
    with timer.timeit("training"):
        losses = train_kge(model, train_cond, ds["num_entities"], cfg, device)
        torch.save(model.state_dict(), outdir / cfg["training"]["checkpoint_name"])

    with timer.timeit("evaluation"):
        filt = ds["train"] + ds["valid"] + ds["test"]
        metrics, top10 = evaluate_tail_only(
            model, ds["test"], filt, ds["num_entities"], ds["orig_degree"], ds["orig_relfreq"], cfg["evaluation"]["max_test_queries"], cfg["evaluation"]["topk"], device
        )

    np.save(outdir / "top10_tail_ids.npy", np.array([x["top10_tail_ids"] for x in top10], dtype=np.int64))
    write_json(
        outdir / "top10_tail_detailed.json",
        {
            "model": cfg["model"]["name"],
            "budget": cfg["perturbation"]["budget"],
            "condition": condition,
            "seed": cfg["seed"],
            "rows": top10,
        },
    )
    write_json(outdir / "metrics.json", metrics)
    write_json(outdir / "perturbation.json", perturb_meta)
    sanity = {"loss_start": losses[0], "loss_end": losses[-1], "roughly_decreasing": losses[-1] < losses[0], "losses": losses}
    write_json(outdir / "training.json", sanity)
    write_json(outdir / "runtime.json", timer.times)
    write_json(outdir / "stability_note.json", {"note": NOTE})


if __name__ == "__main__":
    main()
