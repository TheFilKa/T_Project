from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.data import make_feature_target_split, read_train_test, report_data_quality, validate_schema
from src.evaluate import clip_predictions, score_oof
from src.features import build_feature_frame, make_preprocessor
from src.models import make_model
from src.utils import ensure_dir, load_config, save_joblib, save_json, set_seed
from src.validation import make_folds


def train_one_target(X: pd.DataFrame, y: pd.Series, folds, config: dict, target_name: str):
    models = []
    oof = np.zeros(len(X), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        preprocessor = make_preprocessor(X.iloc[tr_idx], config)
        regressor = make_model(config)
        pipe = Pipeline([("preprocess", preprocessor), ("model", regressor)])

        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = pipe.predict(X.iloc[va_idx])
        models.append(pipe)
        print(f"[{target_name}] fold={fold} done")

    return models, oof


def train(config_path: str = "configs/config.yaml") -> dict:
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    train_df, test_df, sample = read_train_test(config)
    validate_schema(train_df, test_df, config)
    quality = report_data_quality(train_df, test_df, config)

    X_raw, y, _ = make_feature_target_split(train_df, test_df, config)
    X = build_feature_frame(X_raw, config)

    folds = make_folds(X, y, config)
    models_by_target = {}
    oof = pd.DataFrame(index=train_df.index, columns=config["data"]["target_cols"], dtype=float)

    start = time.time()
    for target in config["data"]["target_cols"]:
        models, target_oof = train_one_target(X, y[target], folds, config, target)
        models_by_target[target] = models
        oof[target] = target_oof

    oof = clip_predictions(oof, config)
    scores = score_oof(y, oof, config)

    model_dir = ensure_dir(Path(config["paths"]["models_dir"]) / config["project"]["experiment_name"])
    save_joblib({"models_by_target": models_by_target, "config": config}, model_dir / "models.joblib")
    if config["training"].get("save_oof", True):
        oof.to_csv(model_dir / "oof_predictions.csv", index=False)
    save_json({"scores": scores, "data_quality": quality, "time_sec": time.time() - start}, model_dir / "metrics.json")

    print("CV scores:", scores)
    return {"scores": scores, "model_dir": str(model_dir)}


if __name__ == "__main__":
    train()
