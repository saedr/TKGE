import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


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


def validate_dataset_files(root: str, train: str, valid: str, test: str, allow_fallback: bool = True):
    preferred_root = Path(root)
    preferred = [preferred_root / train, preferred_root / valid, preferred_root / test]
    if all(p.exists() for p in preferred):
        return preferred, str(preferred_root)

    if allow_fallback:
        fallback_root = Path("data")
        fallback = [fallback_root / train, fallback_root / valid, fallback_root / test]
        if all(p.exists() for p in fallback):
            return fallback, str(fallback_root)

    missing = [str(p) for p in preferred if not p.exists()]
    raise FileNotFoundError(f"Required dataset files are missing: {missing}")
