from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def make_folds(X: pd.DataFrame, y: pd.DataFrame, config: dict):
    val_cfg = config["validation"]
    seed = config["project"]["seed"]
    n_splits = val_cfg["n_splits"]
    scheme = val_cfg.get("scheme", "kfold")

    if scheme == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(X, y))

    if scheme == "stratified_bins":
        target = val_cfg.get("target_for_bins", y.columns[0])
        bins = pd.qcut(y[target], q=val_cfg.get("n_bins", 10), labels=False, duplicates="drop")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(X, bins))

    if scheme == "groupkfold":
        group_col = val_cfg.get("group_col")
        if not group_col or group_col not in X.columns:
            raise ValueError("GroupKFold requires validation.group_col present in X")
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X, y, groups=X[group_col]))

    raise ValueError(f"Unknown validation scheme: {scheme}")


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def multi_target_rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    scores = {col: rmse(y_true[col], y_pred[col]) for col in y_true.columns}
    scores["mean"] = float(np.mean(list(scores.values())))
    return scores
