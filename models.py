"""Forecast / classification models and the metrics used in Tables 4-5."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


def make_rf(task: str, n_trees: int, colsample: float, seed: int = 42):
    if task == "reg":
        return RandomForestRegressor(n_trees, max_features="sqrt", n_jobs=-1, random_state=seed)
    return RandomForestClassifier(n_trees, max_features="sqrt", n_jobs=-1, random_state=seed)


def make_lgbm(task: str, n_trees: int, colsample: float, seed: int = 42):
    if lgb is None:
        raise ImportError("lightgbm is not installed")
    cls = lgb.LGBMRegressor if task == "reg" else lgb.LGBMClassifier
    return cls(n_estimators=n_trees, colsample_bytree=colsample, random_state=seed, verbose=-1)


def make_xgb(task: str, n_trees: int, colsample: float, seed: int = 42):
    if xgb is None:
        raise ImportError("xgboost is not installed")
    cls = xgb.XGBRegressor if task == "reg" else xgb.XGBClassifier
    return cls(n_estimators=n_trees, colsample_bytree=colsample, tree_method="hist", random_state=seed, verbosity=0)


# ---------------------------------------------------------------- metrics --
def nd(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return np.abs(p - y).sum() / np.abs(y).sum()


def nrmse(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return np.sqrt(np.mean((p - y) ** 2)) / np.mean(np.abs(y))


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 100 * np.mean(np.abs(y - p) / np.abs(y))


def regression_metrics(y, p) -> dict:
    return {"ND": nd(y, p), "NRMSE": nrmse(y, p), "MAPE": mape(y, p)}


def classification_metrics(y, p, n_classes: int) -> dict:
    labels = list(range(1, n_classes + 1))
    pr, rc, _, _ = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {
        "Accuracy": accuracy_score(y, p),
        "F1-score": f1_score(y, p, average="macro", labels=labels, zero_division=0),
        "Recall": f"{rc.mean():.2f} +/- {rc.std():.2f}",
        "Precision": f"{pr.mean():.2f} +/- {pr.std():.2f}",
    }
