from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation import multi_target_rmse


def apply_si_strategy(pred: pd.DataFrame, config: dict) -> pd.DataFrame:
    pred = pred.copy()
    strategy = config["training"].get("si_strategy", "direct")

    ic50_col = "IC50, mM"
    cc50_col = "CC50, mM"
    si_col = "SI"

    if strategy == "direct":
        return pred

    if strategy == "ratio":
        eps = 1e-9
        pred[si_col] = pred[cc50_col] / pred[ic50_col].clip(lower=eps)
        return pred

    if strategy == "blend":
        eps = 1e-9
        ratio_si = pred[cc50_col] / pred[ic50_col].clip(lower=eps)
        direct_si = pred[si_col]
        pred[si_col] = 0.5 * direct_si + 0.5 * ratio_si
        return pred

    raise ValueError(f"Unknown si_strategy: {strategy}")


def clip_predictions(pred: pd.DataFrame, config: dict) -> pd.DataFrame:
    if config["training"].get("clip_predictions_to_nonnegative", True):
        return pred.clip(lower=0)
    return pred


def score_oof(y_true: pd.DataFrame, oof: pd.DataFrame, config: dict) -> dict:
    pred = clip_predictions(apply_si_strategy(oof, config), config)
    return multi_target_rmse(y_true, pred)
