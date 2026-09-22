"""Lag / calendar features and percentile-based C1-C7 load classes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(s: pd.Series, lags: list[int], step_min: int) -> pd.DataFrame:
    """Lag features + slot-of-day / day-of-week / weekend / month."""
    X = pd.DataFrame({f"lag_{l}": s.shift(l) for l in lags}, index=s.index)
    X["slot"] = (s.index.hour * 60 + s.index.minute) // step_min
    X["dow"] = s.index.dayofweek
    X["is_weekend"] = (X["dow"] >= 5).astype(int)
    X["month"] = s.index.month
    return X


def make_targets(s: pd.Series, h_max: int) -> np.ndarray:
    """Y[i, k] = load at step i + k, for k = 0 .. h_max."""
    return np.column_stack([s.shift(-k).values for k in range(h_max + 1)])


def class_percentiles(q_lmin: float, q_lmax: float, n_charge: int, n_mid: int, n_discharge: int) -> np.ndarray:
    """Cut points (as quantiles in [0, 1]) for the 7 (or N_CLASSES) load classes:
    charge region [0, q_lmin] -> n_charge classes, no-action [q_lmin, q_lmax] -> n_mid,
    discharge [q_lmax, 1] -> n_discharge.
    """
    return np.concatenate([
        np.linspace(0, q_lmin, n_charge + 1),
        np.linspace(q_lmin, q_lmax, n_mid + 1)[1:],
        np.linspace(q_lmax, 1, n_discharge + 1)[1:],
    ])


def class_edges(y: np.ndarray, q_lmin: float, q_lmax: float, n_charge: int, n_mid: int, n_discharge: int) -> np.ndarray:
    """Class boundaries in kW = percentiles of the TRAIN load (no rounding)."""
    edges = np.quantile(y, class_percentiles(q_lmin, q_lmax, n_charge, n_mid, n_discharge))
    edges[0], edges[-1] = 0.0, y.max()
    return edges


def to_class(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map load value(s) in kW to class 1..N_CLASSES using the inner edges."""
    return np.digitize(y, edges[1:-1]) + 1


def balance(c: np.ndarray, mode: str = "max", seed: int = 42) -> np.ndarray:
    """Indices that over/under-sample every class to `mode` ('max' | 'mean') target count.

    ``mode=None`` returns the identity permutation (no resampling).
    """
    rng = np.random.default_rng(seed)
    if mode is None:
        return rng.permutation(len(c))
    classes, counts = np.unique(c, return_counts=True)
    target = counts.max() if mode == "max" else int(round(counts.mean()))
    idx = []
    for cls, n in zip(classes, counts):
        pool = np.where(c == cls)[0]
        idx.append(rng.choice(pool, target, replace=n < target))
    return rng.permutation(np.concatenate(idx))
