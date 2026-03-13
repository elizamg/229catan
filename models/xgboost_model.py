# New file: models/xgboost_model.py

import os
from dataclasses import dataclass
from typing import Any, Optional

import joblib
import numpy as np

from xgboost import XGBRegressor


class IdentityTransformer:
    """Sklearn-like transformer that performs no scaling."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


@dataclass
class XGBConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    gamma: float = 0.0
    n_jobs: int = -1
    random_state: int = 0


def create_model(cfg: XGBConfig) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        gamma=cfg.gamma,
        objective="reg:squarederror",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: XGBConfig,
) -> tuple[XGBRegressor, IdentityTransformer]:
    """Train an XGBoost regressor. Returns (model, scaler).

    Note: trees don't require feature scaling; we return an IdentityTransformer
    to reuse the existing evaluation pipeline that expects a scaler.
    """
    model = create_model(cfg)
    model.fit(X_train, y_train)
    return model, IdentityTransformer()


def save(path: str, model: XGBRegressor, scaler: Optional[Any] = None, cfg: Optional[XGBConfig] = None) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    payload = {
        "model": model,
        "scaler": scaler if scaler is not None else IdentityTransformer(),
        "config": cfg,
        "model_type": "xgb_regressor",
    }
    joblib.dump(payload, path)


def load(path: str) -> tuple[Any, Any, Optional[Any]]:
    payload = joblib.load(path)
    model = payload["model"]
    scaler = payload.get("scaler", IdentityTransformer())
    cfg = payload.get("config")
    return model, scaler, cfg


def top_feature_importances(model: XGBRegressor, k: int = 10) -> np.ndarray:
    """Return indices of top-k features by gain/split importance (uses built-in feature_importances_)."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return np.array([], dtype=int)
    k = min(k, len(importances))
    return np.argsort(importances)[::-1][:k]
