from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def read_train_test(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = config["paths"]
    train = pd.read_csv(paths["train_csv"])
    test = pd.read_csv(paths["test_csv"])
    sample = pd.read_csv(paths["sample_submission_csv"])
    return train, test, sample


def validate_schema(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> None:
    id_col = config["data"]["id_col"]
    targets = config["data"]["target_cols"]

    missing_train = [c for c in [id_col, *targets] if c not in train.columns]
    missing_test = [c for c in [id_col] if c not in test.columns]
    if missing_train:
        raise ValueError(f"Missing train columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing test columns: {missing_test}")

    train_features = set(train.columns) - set(targets)
    test_features = set(test.columns)
    only_train = sorted(train_features - test_features)
    only_test = sorted(test_features - train_features)
    if only_train or only_test:
        raise ValueError(f"Feature schema mismatch. only_train={only_train[:20]}, only_test={only_test[:20]}")


def make_feature_target_split(train: pd.DataFrame, test: pd.DataFrame, config: dict):
    id_col = config["data"]["id_col"]
    targets = config["data"]["target_cols"]
    drop_cols = set(config["data"].get("drop_cols", []))
    drop_cols.update(targets)

    feature_cols = [c for c in train.columns if c not in drop_cols]
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y = train[targets].copy()
    return X_train, y, X_test


def report_data_quality(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> dict:
    targets = config["data"]["target_cols"]
    id_col = config["data"]["id_col"]
    features = [c for c in train.columns if c not in targets]

    si_check = None
    if all(c in train.columns for c in ["IC50, mM", "CC50, mM", "SI"]):
        diff = train["CC50, mM"] / train["IC50, mM"] - train["SI"]
        si_check = {
            "max_abs_error": float(diff.abs().max()),
            "mean_abs_error": float(diff.abs().mean()),
        }

    return {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "n_missing_train": int(train.isna().sum().sum()),
        "n_missing_test": int(test.isna().sum().sum()),
        "duplicated_full_rows_train": int(train.duplicated().sum()),
        "duplicated_ids_train": int(train[id_col].duplicated().sum()),
        "constant_features_train": int((train[features].nunique(dropna=False) <= 1).sum()),
        "target_describe": train[targets].describe().to_dict(),
        "si_ratio_consistency": si_check,
    }
