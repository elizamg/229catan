import os
from dataclasses import dataclass
from typing import Any, Optional

import joblib
import numpy as np
from xgboost import XGBRanker


class IdentityTransformer:
    """Sklearn-like transformer that performs no scaling."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


@dataclass
class XGBRankConfig:
    n_estimators: int = 800
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


def qids_from_game_ids(game_ids: np.ndarray) -> np.ndarray:
    """Map arbitrary game IDs to sequential integer query IDs."""
    mapping: dict[Any, int] = {}
    out = np.empty(len(game_ids), dtype=np.uint32)
    next_id = 0

    for i, gid in enumerate(game_ids):
        if gid not in mapping:
            mapping[gid] = next_id
            next_id += 1
        out[i] = mapping[gid]

    return out


def create_model(cfg: XGBRankConfig) -> XGBRanker:
    return XGBRanker(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        gamma=cfg.gamma,
        objective="rank:pairwise",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )


def train_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    game_ids_train: np.ndarray,
    cfg: XGBRankConfig,
) -> tuple[XGBRanker, IdentityTransformer]:
    """Train an XGBoost ranker grouped by game_id. Returns (model, scaler)."""
    model = create_model(cfg)
    qid = qids_from_game_ids(game_ids_train)
    model.fit(X_train, y_train, qid=qid)
    return model, IdentityTransformer()


def save(path: str, model: XGBRanker, scaler: Optional[Any] = None, cfg: Optional[XGBRankConfig] = None) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    payload = {
        "model": model,
        "scaler": scaler if scaler is not None else IdentityTransformer(),
        "config": cfg,
        "model_type": "xgb_ranker",
    }
    joblib.dump(payload, path)


def load(path: str) -> tuple[Any, Any, Optional[Any]]:
    payload = joblib.load(path)
    model = payload["model"]
    scaler = payload.get("scaler", IdentityTransformer())
    cfg = payload.get("config")
    return model, scaler, cfg
