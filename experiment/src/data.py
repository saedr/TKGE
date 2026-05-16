from collections import Counter
from pathlib import Path

from .utils import write_json, validate_dataset_files


def _read_tsv(path: Path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            triples.append(tuple(parts))
    return triples


def load_dataset(cfg: dict, mapping_out_path: str):
    dcfg = cfg["dataset"]
    (train_p, valid_p, test_p), dataset_root_used = validate_dataset_files(dcfg["root"], dcfg["train"], dcfg["valid"], dcfg["test"])

    train_str = _read_tsv(train_p)
    valid_str = _read_tsv(valid_p)
    test_str = _read_tsv(test_p)

    entities = sorted({e for h, _, t in (train_str + valid_str + test_str) for e in (h, t)})
    relations = sorted({r for _, r, _ in (train_str + valid_str + test_str)})
    e2id = {e: i for i, e in enumerate(entities)}
    r2id = {r: i for i, r in enumerate(relations)}

    def encode(triples):
        return [(e2id[h], r2id[r], e2id[t]) for h, r, t in triples]

    out = {
        "entity_count": len(e2id),
        "relation_count": len(r2id),
        "train_count": len(train_str),
        "valid_count": len(valid_str),
        "test_count": len(test_str),
        "dataset_root_used": dataset_root_used,
    }
    write_json(mapping_out_path, out)

    train = encode(train_str)
    valid = encode(valid_str)
    test = encode(test_str)

    deg = Counter()
    relfreq = Counter()
    for h, r, t in train:
        deg[h] += 1
        deg[t] += 1
        relfreq[r] += 1

    return {
        "train": train,
        "valid": valid,
        "test": test,
        "num_entities": len(e2id),
        "num_relations": len(r2id),
        "orig_degree": deg,
        "orig_relfreq": relfreq,
    }
