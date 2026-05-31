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
        ic50_floor = config["training"].get("ic50_floor_for_si", 1.0)
        pred[si_col] = pred[cc50_col] / pred[ic50_col].clip(lower=ic50_floor)
        return pred

    if strategy == "blend":
        ic50_floor = config["training"].get("ic50_floor_for_si", 1e-9)
        alpha = config["training"].get("si_blend_alpha", 0.5)

        ratio_si = pred[cc50_col] / pred[ic50_col].clip(lower=ic50_floor)
        direct_si = pred[si_col]

        pred[si_col] = alpha * direct_si + (1.0 - alpha) * ratio_si
        return pred

    raise ValueError(f"Unknown si_strategy: {strategy}")


def clip_predictions(pred: pd.DataFrame, config: dict) -> pd.DataFrame:
    if config["training"].get("clip_predictions_to_nonnegative", True):
        return pred.clip(lower=0)
    return pred


def score_oof(y_true: pd.DataFrame, oof: pd.DataFrame, config: dict) -> dict:
    pred = clip_predictions(apply_si_strategy(oof, config), config)
    return multi_target_rmse(y_true, pred)
