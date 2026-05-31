from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import make_feature_target_split, read_train_test, validate_schema
from src.evaluate import apply_si_strategy, clip_predictions
from src.features import build_feature_frame
from src.utils import ensure_dir, load_config, load_joblib


def predict(config_path: str = "configs/config.yaml", model_path: str | None = None) -> str:
    config = load_config(config_path)
    train_df, test_df, sample = read_train_test(config)
    validate_schema(train_df, test_df, config)

    _, _, X_test_raw = make_feature_target_split(train_df, test_df, config)
    X_test = build_feature_frame(X_test_raw, config)

    if model_path is None:
        model_path = Path(config["paths"]["models_dir"]) / config["project"]["experiment_name"] / "models.joblib"
    artifact = load_joblib(model_path)
    models_by_target = artifact["models_by_target"]

    pred = pd.DataFrame(index=test_df.index)
    for target in config["data"]["target_cols"]:
        fold_preds = np.column_stack([m.predict(X_test) for m in models_by_target[target]])
        pred[target] = fold_preds.mean(axis=1)

    pred = clip_predictions(apply_si_strategy(pred, config), config)

    submission = sample[[config["data"]["id_col"]]].copy()
    for train_target, sub_target in zip(config["data"]["target_cols"], config["data"]["submission_target_cols"]):
        submission[sub_target] = pred[train_target].values

    out_dir = ensure_dir(config["paths"]["submissions_dir"])
    out_path = out_dir / f"{config['project']['experiment_name']}_submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    predict()
