import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


REQUIRED_ERROR = "FB15k-237 files are required. Place train.txt, valid.txt, and test.txt at data/FB15k-237/."


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class StageTimer:
    def __init__(self):
        self.times = {}

    def timeit(self, name):
        class _Ctx:
            def __enter__(_self):
                _self.start = time.time()

            def __exit__(_self, exc_type, exc, tb):
                self.times[name] = self.times.get(name, 0.0) + (time.time() - _self.start)

        return _Ctx()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, obj: dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_dataset_files(root: str, train: str, valid: str, test: str):
    files = [Path(root) / train, Path(root) / valid, Path(root) / test]
    if not all(p.exists() for p in files):
        raise FileNotFoundError(REQUIRED_ERROR)
    return files
