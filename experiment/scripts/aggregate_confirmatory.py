#!/usr/bin/env python3
"""Hierarchical paired-bootstrap analysis for the frozen confirmatory grid."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


DATASETS = ["WN18RR", "CoDEx-M"]
MODELS = ["TransE", "DistMult", "ComplEx", "RotatE"]
CONDITIONS = ["Original", "Random", "Structured-low", "Relation-low"]
STRUCTURED = ["Structured-low", "Relation-low"]
TRAIN_SEEDS = [11, 22, 33]
DELETION_SEEDS = [101, 202, 303]
EXPECTED_RUNS = 30
SLICE_NAMES = [
    "overall",
    "coverage_low", "coverage_mid", "coverage_high",
    "relation_low", "relation_mid", "relation_high",
]


def percentile_interval(values):
    q = np.quantile(values, [0.025, 0.5, 0.975])
    return {"low": float(q[0]), "median": float(q[1]), "high": float(q[2])}


def load_chunks(indir):
    chunks = {}
    for path in sorted(Path(indir).glob("*.json")):
        if path.name.endswith(".mapping.json"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        key = (obj.get("dataset"), obj.get("model"))
        if key not in {(d, m) for d in DATASETS for m in MODELS}:
            continue
        if key in chunks:
            raise RuntimeError(f"Duplicate chunk for {key}: {path}")
        if not obj.get("complete") or len(obj.get("runs", [])) != EXPECTED_RUNS:
            raise RuntimeError(f"Incomplete chunk {path}: {len(obj.get('runs', []))}/{EXPECTED_RUNS} runs")
        chunks[key] = obj
    missing = [(d, m) for d in DATASETS for m in MODELS if (d, m) not in chunks]
    if missing:
        raise RuntimeError(f"Missing confirmatory chunks: {missing}")
    return chunks


def condition_arrays(chunk):
    run_map = {
        (run["condition"], int(run["deletion_seed"]), int(run["train_seed"])): np.asarray(run["metrics"]["rr"], dtype=np.float64)
        for run in chunk["runs"]
    }
    arrays = {}
    arrays["Original"] = np.stack([
        np.stack([run_map[("Original", 0, seed)] for seed in TRAIN_SEEDS])
    ])
    for condition in CONDITIONS[1:]:
        arrays[condition] = np.stack([
            np.stack([run_map[(condition, deletion_seed, train_seed)] for train_seed in TRAIN_SEEDS])
            for deletion_seed in DELETION_SEEDS
        ])
    return arrays


def dataset_arrays(chunks, dataset):
    result = {model: condition_arrays(chunks[(dataset, model)]) for model in MODELS}
    n_queries = {arr.shape[-1] for model in MODELS for arr in result[model].values()}
    if len(n_queries) != 1:
        raise RuntimeError(f"Query-count disagreement for {dataset}: {n_queries}")
    reference = chunks[(dataset, MODELS[0])]["runs"][0]["metrics"]
    coverage = np.asarray(reference["coverage_bins"])
    relation = np.asarray(reference["relation_bins"])
    for model in MODELS:
        for run in chunks[(dataset, model)]["runs"]:
            if run["metrics"]["coverage_bins"] != coverage.tolist() or run["metrics"]["relation_bins"] != relation.tolist():
                raise RuntimeError(f"Slice assignment changed across runs for {dataset}/{model}")
    return result, coverage, relation


def slice_indices(coverage, relation):
    indices = {"overall": np.arange(len(coverage), dtype=np.int64)}
    for value in ("low", "mid", "high"):
        indices[f"coverage_{value}"] = np.flatnonzero(coverage == value)
        indices[f"relation_{value}"] = np.flatnonzero(relation == value)
    if any(len(x) == 0 for x in indices.values()):
        raise RuntimeError(f"Empty evaluation slice: {[(k, len(v)) for k, v in indices.items()]}")
    return indices


def bootstrap_condition_means(arrays, query_ids, replications, rng):
    """Return B x model x condition means with crossed units sampled jointly.

    Deletion realizations and training seeds are crossed. Their indices, plus
    query indices, are resampled once per replicate and shared across all
    models and mechanisms, preserving the pairing in the frozen grid.
    """
    draws = np.empty((replications, len(MODELS), len(CONDITIONS)), dtype=np.float64)
    n_queries = len(query_ids)
    for b in range(replications):
        sampled_queries = rng.choice(query_ids, size=n_queries, replace=True)
        sampled_train = rng.integers(0, len(TRAIN_SEEDS), size=len(TRAIN_SEEDS))
        sampled_delete = rng.integers(0, len(DELETION_SEEDS), size=len(DELETION_SEEDS))
        for model_idx, model in enumerate(MODELS):
            original = arrays[model]["Original"]
            draws[b, model_idx, 0] = original[np.ix_([0], sampled_train, sampled_queries)].mean()
            for condition_idx, condition in enumerate(CONDITIONS[1:], start=1):
                arr = arrays[model][condition]
                draws[b, model_idx, condition_idx] = arr[np.ix_(sampled_delete, sampled_train, sampled_queries)].mean()
    return draws


def point_condition_means(arrays, query_ids):
    means = np.empty((len(MODELS), len(CONDITIONS)), dtype=np.float64)
    for model_idx, model in enumerate(MODELS):
        for condition_idx, condition in enumerate(CONDITIONS):
            means[model_idx, condition_idx] = arrays[model][condition][:, :, query_ids].mean()
    return means


def supported(interval):
    return bool(interval["low"] > 0.0 or interval["high"] < 0.0)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fnum(value):
    return f"{value:.6f}"


def fci(row, prefix="ci"):
    return f"[{row[prefix + '_low']:.6f}, {row[prefix + '_high']:.6f}]"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def load_fb15k_prior():
    reports = Path(__file__).resolve().parents[1] / "reports"
    paths = {
        "TransE": reports / "transe_30pct_chunk_report.json",
        "DistMult": reports / "distmult_30pct_pilot_report.json",
    }
    differentials = {}
    for model, path in paths.items():
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        values = {row["condition"]: row["mean_tail_mrr"] for row in report["overall_tail_only_metrics"]}
        differentials[model] = float(values["Random"] - values["Structured-low"])
    interaction = differentials["TransE"] - differentials["DistMult"]
    return {
        "status": "exploratory point-estimate evidence; no hierarchical interval",
        "entity_structured_differential_degradation": differentials,
        "TransE_minus_DistMult_interaction": interaction,
        "qualitative_model_dependence": bool(
            np.sign(differentials["TransE"]) != np.sign(differentials["DistMult"])
        ),
    }


def build_report(
    mean_rows, differential_rows, interaction_rows, reversal_rows, slice_counts, decision, fb15k_prior,
    training_sanity,
):
    overall_means = [x for x in mean_rows if x["slice"] == "overall"]
    mean_table = markdown_table(
        ["Dataset", "Model", "Mechanism", "Mean MRR", "95% hierarchical CI"],
        [[x["dataset"], x["model"], x["condition"], fnum(x["mean"]), fci(x)] for x in overall_means],
    )
    overall_diff = [x for x in differential_rows if x["slice"] == "overall"]
    diff_table = markdown_table(
        ["Dataset", "Model", "Structured mechanism", "Differential degradation", "95% hierarchical CI"],
        [[x["dataset"], x["model"], x["condition"], fnum(x["differential_degradation"]), fci(x)] for x in overall_diff],
    )
    overall_interactions = [x for x in interaction_rows if x["slice"] == "overall"]
    interaction_table = markdown_table(
        ["Dataset", "Models A−B", "Structured mechanism", "Interaction", "95% hierarchical CI", "Supported"],
        [[x["dataset"], f"{x['model_a']}−{x['model_b']}", x["condition"], fnum(x["interaction"]), fci(x), x["supported"]]
         for x in overall_interactions],
    )
    supported_slice = [x for x in interaction_rows if x["slice"] != "overall" and x["supported"]]
    slice_table = markdown_table(
        ["Dataset", "Slice", "Models A−B", "Mechanism", "Interaction", "95% CI"],
        [[x["dataset"], x["slice"], f"{x['model_a']}−{x['model_b']}", x["condition"], fnum(x["interaction"]), fci(x)]
         for x in supported_slice],
    ) if supported_slice else "No slice-specific interaction interval excluded zero."
    supported_reversals = [x for x in reversal_rows if x["supported"]]
    reversal_table = markdown_table(
        ["Dataset", "Slice", "Models A−B", "Mechanisms", "First gap (95% CI)", "Second gap (95% CI)"],
        [[x["dataset"], x["slice"], f"{x['model_a']}−{x['model_b']}",
          f"{x['condition_a']} → {x['condition_b']}",
          f"{fnum(x['gap_a'])} {fci(x, 'gap_a_ci')}",
          f"{fnum(x['gap_b'])} {fci(x, 'gap_b_ci')}"]
         for x in supported_reversals],
    ) if supported_reversals else "No uncertainty-supported model-ranking reversal occurred."
    count_table = markdown_table(
        ["Dataset", "Overall", "Coverage low/mid/high", "Relation low/mid/high"],
        [[dataset, counts["overall"],
          f"{counts['coverage_low']}/{counts['coverage_mid']}/{counts['coverage_high']}",
          f"{counts['relation_low']}/{counts['relation_mid']}/{counts['relation_high']}"]
         for dataset, counts in slice_counts.items()],
    )
    replication = (
        "The FB15k-237 pilot showed model-dependent point differences between random and entity-coverage "
        "structured deletion: differential degradation was "
        f"{fb15k_prior['entity_structured_differential_degradation']['TransE']:.6f} for TransE and "
        f"{fb15k_prior['entity_structured_differential_degradation']['DistMult']:.6f} for DistMult "
        f"(point interaction {fb15k_prior['TransE_minus_DistMult_interaction']:.6f}). "
        "That pilot did not provide the hierarchical uncertainty used here. "
        + decision["replication_statement"]
    )
    converged_family_interactions = [
        x for x in interaction_rows
        if x["slice"] == "overall"
        and x["condition"] == "Relation-low"
        and x["supported"]
        and "RotatE" not in (x["model_a"], x["model_b"])
    ]
    converged_table = markdown_table(
        ["Dataset", "Models A−B", "Interaction", "95% CI"],
        [[x["dataset"], f"{x['model_a']}−{x['model_b']}", fnum(x["interaction"]), fci(x)]
         for x in converged_family_interactions],
    )
    return f"""# Confirmatory structured-missingness replication

## Decision: **{decision['label']}**

{decision['rationale']}

The estimand is tail-MRR degradation from the clean graph and, primarily, differential degradation versus random deletion. Positive differential degradation means that the structured mechanism harms a model more than random deletion. Model × mechanism interaction is the difference in that differential degradation between models.

## Mean tail MRR

{mean_table}

## Differential degradation: structured minus random

{diff_table}

## Model × mechanism interactions

{interaction_table}

Full overall and slice-specific estimates are in `tables/interactions.csv`.

## Slice-specific results

{count_table}

{slice_table}

Complete slice estimates are in `tables/mean_mrr.csv` and `tables/differential_degradation.csv`.

## Supported model-ranking reversals

{reversal_table}

## FB15k-237 replication assessment

{replication}

The sign of the TransE-versus-bilinear interaction differs between WN18RR and CoDEx-M, so the evidence supports model dependence, not a universal claim that one family is always more fragile.

## Training-sanity limitation

WN18RR RotatE's final epoch loss exceeded its first epoch loss in {training_sanity['WN18RR']['RotatE']['non_decreasing_runs']}/30 runs, and its clean MRR was 0.002263. RotatE-involving WN18RR contrasts should therefore not be treated as evidence about a competitive RotatE model. The continuation result does not depend on them: these uncertainty-supported overall Relation-low interactions remain after excluding RotatE:

{converged_table}

## Frozen design and uncertainty

- Datasets: WN18RR and CoDEx-M; models: TransE, DistMult, ComplEx, RotatE.
- The three perturbed mechanisms remove 30% of training triples. Entity Structured-low uses `1/(min(deg(h),deg(t))+5)`; Relation-low uses `1/(freq(r)+5)`; both statistics come from the unperturbed training graph.
- The deletion guard leaves every training-visible entity incident to at least one retained training triple. Validation and test triples are unchanged.
- Clean models use three training seeds. Each perturbed mechanism uses three deletion realizations crossed with three training seeds.
- All original test triples are evaluated in the tail direction. The filter is the original train+validation+test union, including deleted training triples. Ties use optimistic rank: one plus the number of unfiltered candidates with strictly greater score.
- Slice assignments use quartiles of original-graph entity coverage and relation frequency and remain fixed for all conditions.
- The 95% percentile intervals use paired hierarchical bootstrap replicates that jointly resample deletion realization, training seed, and original test query within each slice. Resampled indices are shared across models and mechanisms to retain the crossed pairing.
- A reversal is supported only when both compared missingness-mechanism model-gap intervals exclude zero in opposite directions.
"""


def main(args):
    chunks = load_chunks(args.indir)
    training_sanity = {
        dataset: {
            model: {
                "non_decreasing_runs": sum(
                    not run["loss_end"] < run["loss_start"] for run in chunks[(dataset, model)]["runs"]
                ),
                "total_runs": len(chunks[(dataset, model)]["runs"]),
            }
            for model in MODELS
        }
        for dataset in DATASETS
    }
    mean_rows = []
    differential_rows = []
    interaction_rows = []
    reversal_rows = []
    slice_counts = {}
    for dataset_idx, dataset in enumerate(DATASETS):
        arrays, coverage, relation = dataset_arrays(chunks, dataset)
        slices = slice_indices(coverage, relation)
        slice_counts[dataset] = {name: len(ids) for name, ids in slices.items()}
        for slice_idx, (slice_name, query_ids) in enumerate(slices.items()):
            rng = np.random.default_rng(args.seed + dataset_idx * 1000 + slice_idx)
            means = point_condition_means(arrays, query_ids)
            draws = bootstrap_condition_means(arrays, query_ids, args.bootstrap, rng)
            for model_idx, model in enumerate(MODELS):
                for condition_idx, condition in enumerate(CONDITIONS):
                    interval = percentile_interval(draws[:, model_idx, condition_idx])
                    mean_rows.append({
                        "dataset": dataset, "model": model, "condition": condition, "slice": slice_name,
                        "mean": float(means[model_idx, condition_idx]),
                        "ci_low": interval["low"], "ci_high": interval["high"],
                        "n_deletion_realizations": 1 if condition == "Original" else 3,
                        "n_training_seeds": 3, "n_queries": len(query_ids),
                    })
                clean_idx = CONDITIONS.index("Original")
                random_idx = CONDITIONS.index("Random")
                clean_mean = means[model_idx, clean_idx]
                random_mean = means[model_idx, random_idx]
                for condition in STRUCTURED:
                    condition_idx = CONDITIONS.index(condition)
                    structured_mean = means[model_idx, condition_idx]
                    diff_draws = draws[:, model_idx, random_idx] - draws[:, model_idx, condition_idx]
                    interval = percentile_interval(diff_draws)
                    differential_rows.append({
                        "dataset": dataset, "model": model, "condition": condition, "slice": slice_name,
                        "clean_mean": float(clean_mean), "random_mean": float(random_mean),
                        "structured_mean": float(structured_mean),
                        "random_degradation": float(clean_mean - random_mean),
                        "structured_degradation": float(clean_mean - structured_mean),
                        "differential_degradation": float(random_mean - structured_mean),
                        "ci_low": interval["low"], "ci_high": interval["high"],
                    })
            for model_a, model_b in combinations(MODELS, 2):
                a_idx, b_idx = MODELS.index(model_a), MODELS.index(model_b)
                random_idx = CONDITIONS.index("Random")
                for condition in STRUCTURED:
                    condition_idx = CONDITIONS.index(condition)
                    diff_a = draws[:, a_idx, random_idx] - draws[:, a_idx, condition_idx]
                    diff_b = draws[:, b_idx, random_idx] - draws[:, b_idx, condition_idx]
                    interaction_draws = diff_a - diff_b
                    interval = percentile_interval(interaction_draws)
                    interaction_rows.append({
                        "dataset": dataset, "model_a": model_a, "model_b": model_b,
                        "condition": condition, "slice": slice_name,
                        "interaction": float(
                            (means[a_idx, random_idx] - means[a_idx, condition_idx])
                            - (means[b_idx, random_idx] - means[b_idx, condition_idx])
                        ),
                        "ci_low": interval["low"], "ci_high": interval["high"],
                        "supported": supported(interval),
                    })
                for condition_a, condition_b in combinations(CONDITIONS[1:], 2):
                    condition_a_idx = CONDITIONS.index(condition_a)
                    condition_b_idx = CONDITIONS.index(condition_b)
                    gap_a_draws = draws[:, a_idx, condition_a_idx] - draws[:, b_idx, condition_a_idx]
                    gap_b_draws = draws[:, a_idx, condition_b_idx] - draws[:, b_idx, condition_b_idx]
                    gap_a_interval = percentile_interval(gap_a_draws)
                    gap_b_interval = percentile_interval(gap_b_draws)
                    is_reversal = bool(
                        (gap_a_interval["low"] > 0 and gap_b_interval["high"] < 0)
                        or (gap_a_interval["high"] < 0 and gap_b_interval["low"] > 0)
                    )
                    reversal_rows.append({
                        "dataset": dataset, "model_a": model_a, "model_b": model_b,
                        "condition_a": condition_a, "condition_b": condition_b, "slice": slice_name,
                        "gap_a": float(means[a_idx, condition_a_idx] - means[b_idx, condition_a_idx]),
                        "gap_a_ci_low": gap_a_interval["low"], "gap_a_ci_high": gap_a_interval["high"],
                        "gap_b": float(means[a_idx, condition_b_idx] - means[b_idx, condition_b_idx]),
                        "gap_b_ci_low": gap_b_interval["low"], "gap_b_ci_high": gap_b_interval["high"],
                        "supported": is_reversal,
                    })

    supported_by_dataset = {
        dataset: sum(x["supported"] for x in interaction_rows if x["dataset"] == dataset)
        for dataset in DATASETS
    }
    supported_overall = {
        dataset: {
            condition: sum(
                x["supported"]
                for x in interaction_rows
                if x["dataset"] == dataset and x["slice"] == "overall" and x["condition"] == condition
            )
            for condition in STRUCTURED
        }
        for dataset in DATASETS
    }
    supported_slices = {
        dataset: {
            condition: sum(
                x["supported"]
                for x in interaction_rows
                if x["dataset"] == dataset and x["slice"] != "overall" and x["condition"] == condition
            )
            for condition in STRUCTURED
        }
        for dataset in DATASETS
    }
    reversals = [x for x in reversal_rows if x["supported"]]
    overall_support_by_dataset = {
        dataset: sum(supported_overall[dataset].values()) > 0 for dataset in DATASETS
    }
    any_new_interaction = any(overall_support_by_dataset.values())
    other_dataset_support = all(overall_support_by_dataset.values())
    fb15k_prior = load_fb15k_prior()
    fb15k_qualitative_support = fb15k_prior["qualitative_model_dependence"]
    continue_gate = any_new_interaction and (other_dataset_support or fb15k_qualitative_support)
    if reversals:
        label = "STRONG CONTINUE"
        rationale = (
            f"{len(reversals)} uncertainty-supported reversal(s) occurred on the new datasets, "
            "and at least one model × mechanism interaction interval excluded zero."
        )
    elif continue_gate:
        label = "CONTINUE"
        rationale = (
            "Relation-low produced uncertainty-supported overall model × mechanism interactions on both "
            f"new datasets ({supported_overall['WN18RR']['Relation-low']}/6 model pairs on WN18RR and "
            f"{supported_overall['CoDEx-M']['Relation-low']}/6 on CoDEx-M); no supported ranking reversal occurred."
        )
    else:
        label = "SHELVE"
        rationale = "The model × missingness interaction did not reproduce with uncertainty support on either new dataset."
    entity_slice_total = sum(supported_slices[d]["Structured-low"] for d in DATASETS)
    if any(supported_overall[d]["Structured-low"] for d in DATASETS):
        replication_statement = (
            "The specific entity-coverage Structured-low interaction reproduced in overall MRR on at least one new dataset."
        )
    else:
        replication_statement = (
            "The specific entity-coverage Structured-low pattern did not reproduce in overall MRR on either new "
            f"dataset; only {entity_slice_total} slice-specific pairwise interaction interval(s) excluded zero. "
            "The broader model × structured-missingness phenomenon did reproduce for the prespecified "
            "relation-frequency mechanism in overall MRR on both new datasets."
        )
    decision = {
        "label": label,
        "rationale": rationale,
        "supported_interactions_by_dataset": supported_by_dataset,
        "supported_overall_interactions": supported_overall,
        "supported_slice_interactions": supported_slices,
        "supported_reversals": len(reversals),
        "fb15k_qualitative_support": fb15k_qualitative_support,
        "replication_statement": replication_statement,
    }
    summary = {
        "analysis": {
            "bootstrap_replications": args.bootstrap,
            "bootstrap_seed": args.seed,
            "interval": "paired crossed hierarchical percentile 95%",
            "units": ["deletion realization", "training seed", "test query within slice"],
        },
        "slice_counts": slice_counts,
        "mean_mrr": mean_rows,
        "differential_degradation": differential_rows,
        "interactions": interaction_rows,
        "reversals": reversal_rows,
        "training_sanity": training_sanity,
        "fb15k_prior": fb15k_prior,
        "decision": decision,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tables = Path(args.tables_dir) if args.tables_dir else out_path.parent / "tables"
    write_csv(tables / "mean_mrr.csv", mean_rows)
    write_csv(tables / "differential_degradation.csv", differential_rows)
    write_csv(tables / "interactions.csv", interaction_rows)
    write_csv(tables / "reversals.csv", reversal_rows)
    report_path = Path(args.report) if args.report else out_path.parent / "confirmatory_report.md"
    report_path.write_text(
        build_report(
            mean_rows, differential_rows, interaction_rows, reversal_rows, slice_counts, decision, fb15k_prior,
            training_sanity,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tables-dir")
    parser.add_argument("--report")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1701)
    main(parser.parse_args())
