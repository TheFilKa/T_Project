from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_joblib(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)
