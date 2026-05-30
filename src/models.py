from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import numpy as np


def make_base_model(config: dict):
    name = config["training"]["model_name"]
    params = config["models"].get(name, {}).copy()

    if name == "dummy":
        return DummyRegressor(strategy="median")
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(**params)
    if name == "extratrees":
        return ExtraTreesRegressor(**params)
    if name == "histgb":
        return HistGradientBoostingRegressor(**params)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", **params)
    raise ValueError(f"Unknown model_name: {name}")


def maybe_wrap_target_transform(model, config: dict):
    transform = config["training"].get("target_transform", "none")
    if transform == "none":
        return model
    if transform == "log1p":
        return TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    raise ValueError(f"Unknown target_transform: {transform}")


def make_model(config: dict):
    return maybe_wrap_target_transform(make_base_model(config), config)
