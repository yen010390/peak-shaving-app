"""Algorithm 1 (forecast-based peak shaving) and the billing cost model."""
from __future__ import annotations

import numpy as np
import pandas as pd


def peak_shaving(y1: np.ndarray, u: np.ndarray, kind: str, thr: tuple[float, float],
                  delta: float, soc_max: float, soc_min: float = 0.0, scale: float = 1.0):
    """Algorithm 1.

    y1: one-step forecast (kW if kind='reg', class if kind='clf').
    u : (T, H) forecast of the next H steps (same units as y1).
    thr: (low, high) thresholds -- (L_min, L_max) for 'reg', (charge_max_class, discharge_min_class) for 'clf'.

        y <= low  and SOC < SOC_max                      -> CHARGE:    SOC <- min(SOC_max, SOC + delta)
        y >= high and SOC > SOC_min and floor(SOC/delta) > eta -> DISCHARGE: SOC <- max(SOC_min, SOC - delta)
        (eta = number of the next H steps forecast higher than y)

    SOC is tracked as an INTEGER number of delta units to avoid floating-point
    error (e.g. 25.2 - 3*6.3 = 6.2999... < delta).  Returns (action, soc_kw)
    where action in {+1 charge, -1 discharge, 0 nothing}.
    """
    lo, hi = thr
    d = delta * scale
    soc_max_u = int(round(soc_max * scale / d))
    soc_min_u = int(round(soc_min * scale / d))
    soc = soc_max_u  # battery starts full
    n = len(y1)
    A = np.zeros(n, dtype=int)
    soc_log = np.zeros(n)
    for t in range(n):
        y = y1[t]
        if y <= lo:
            if soc < soc_max_u:
                A[t] = 1
                soc = min(soc_max_u, soc + 1)
        elif y >= hi:
            if soc > soc_min_u:
                eta = int(np.sum(u[t] > y))
                if soc > eta:
                    A[t] = -1
                    soc = max(soc_min_u, soc - 1)
        soc_log[t] = soc * d
    return A, soc_log


def apply_actions(net_load: np.ndarray, action: np.ndarray, delta: float, scale: float = 1.0) -> np.ndarray:
    """Net load seen by the grid after charge (+delta) / discharge (-delta, capped)."""
    d = delta * scale
    out = net_load.astype(float).copy()
    out[action == 1] += d
    dis = action == -1
    out[dis] = np.maximum(out[dis] - d, 0.0)
    return out


def daily_cost(net: np.ndarray, index: pd.DatetimeIndex, on_peak_hours, rate_off: float, rate_on: float,
                rate_peak: float, dt_hours: float, steps_per_day: int, e_mult: float = 1.0, p_mult: float = 1.0,
                eval_dates: pd.DatetimeIndex | None = None):
    """Per-day total cost (energy + peak charge), for full days (and, if given, only
    days in ``eval_dates``). Returns (total, peak_charge) as pd.Series indexed by day.
    """
    rate = np.where(index.hour.isin(list(on_peak_hours)), rate_on, rate_off) * e_mult
    day = index.normalize()
    energy = pd.Series(net * dt_hours * rate, index=index).groupby(day).sum()
    peak = pd.Series(net, index=index).groupby(day).max() * rate_peak * p_mult
    full = pd.Series(1, index=index).groupby(day).sum() == steps_per_day
    keep = full if eval_dates is None else (full & full.index.isin(eval_dates))
    return (energy + peak)[keep], peak[keep]
