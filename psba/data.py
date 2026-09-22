"""Load, validate and clean a load (kW) time series from a CSV file."""
from __future__ import annotations

import numpy as np
import pandas as pd


def infer_granularity(raw: pd.DataFrame, time_col: str) -> pd.Timedelta:
    """Delta t = mode(t[i+1] - t[i]) over the (deduplicated, sorted) timestamps."""
    ts = pd.to_datetime(raw[time_col], errors="coerce").dropna().drop_duplicates().sort_values()
    gaps = ts.diff().dropna()
    if len(gaps) == 0:
        raise ValueError("Not enough valid timestamps to infer granularity.")
    return gaps.mode().iloc[0]


def time_grid(freq: pd.Timedelta) -> dict:
    step_min = int(freq / pd.Timedelta(minutes=1))
    dt_hours = freq / pd.Timedelta(hours=1)
    steps_per_day = int(pd.Timedelta(days=1) // freq)
    lags = sorted(set(range(1, 25)) | {steps_per_day, 7 * steps_per_day})
    return dict(freq=freq, step_min=step_min, dt_hours=dt_hours, steps_per_day=steps_per_day, lags=lags)


def spike_flags(s: pd.Series, window: int = 9, k: float = 6.0) -> pd.Series:
    """Isolated spikes: |value - local median| > k*MAD, jump vs. BOTH neighbours, same sign.

    A load peak that lasts several steps is *not* flagged — only a single
    sample that jumps away from its neighbours and back.
    """
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    scale = (1.4826 * mad).clip(lower=0.05 * s.abs().median())
    up, dn = s - s.shift(1), s - s.shift(-1)
    return (
        ((s - med).abs() > k * scale)
        & (up.abs() > k * scale)
        & (dn.abs() > k * scale)
        & (np.sign(up) == np.sign(dn))
    )


def check_data(raw: pd.DataFrame, time_col: str, val_col: str, freq: pd.Timedelta,
                dt_hours: float, steps_per_day: int) -> pd.DataFrame:
    """Data-quality report: count of each issue type and a PASS/WARN status."""
    t = pd.to_datetime(raw[time_col], errors="coerce")
    v = pd.to_numeric(raw[val_col], errors="coerce")
    x = pd.Series(v.values, index=t)[t.notna().values]
    x = x[~x.index.duplicated()].sort_index()
    x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq=freq))
    run = (x != x.shift()).cumsum()
    days = x.dropna().groupby(x.dropna().index.normalize()).size()
    spikes = int(spike_flags(x).sum())
    counts = {
        "unparsable timestamp": int(t.isna().sum()),
        "non-numeric load": int(v.isna().sum() - raw[val_col].isna().sum()),
        "duplicate timestamp": int(t.duplicated().sum()),
        "not sorted by time": int(not t.is_monotonic_increasing),
        "missing / empty steps": int(x.isna().sum()),
        "negative values": int((x < 0).sum()),
        "zero values": int((x == 0).sum()),
        "stuck (constant) run >= 12h": int(run.groupby(run).size().max() >= 12 / dt_hours),
        f"days with < {steps_per_day} samples": int((days < steps_per_day).sum()),
        "isolated spikes": spikes,
    }
    rep = pd.DataFrame({"count": counts})
    ok = rep["count"] == 0
    ok.loc["isolated spikes"] = spikes <= 0.0005 * len(x)  # allow up to 0.05%
    rep["status"] = np.where(ok, "PASS", "WARN")
    return rep


def clean_data(raw: pd.DataFrame, time_col: str, val_col: str, freq: pd.Timedelta,
                dt_hours: float, step_min: int) -> pd.DataFrame:
    """Merge duplicates, build a regular index, drop negatives, fill short gaps.

    Spikes are only *flagged* (``is_spike``), never altered, so real peaks
    are preserved.
    """
    d = pd.DataFrame({
        time_col: pd.to_datetime(raw[time_col], errors="coerce"),
        val_col: pd.to_numeric(raw[val_col], errors="coerce"),
    })
    s = d.dropna(subset=[time_col]).groupby(time_col)[val_col].mean().sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=freq))
    s[s < 0] = np.nan
    max_gap = max(1, round(1 / dt_hours))  # fill at most 1 hour of consecutive gaps
    out = pd.DataFrame({val_col: s, "is_spike": spike_flags(s), "is_imputed": s.isna()})
    s = s.interpolate("time", limit=max_gap, limit_area="inside")
    profile = s.groupby([s.index.dayofweek, (s.index.hour * 60 + s.index.minute) // step_min]).transform("median")
    out[val_col] = s.fillna(profile).ffill().bfill()
    out.index.name = time_col
    return out


def load_and_clean(raw: pd.DataFrame, time_col: str = "datetime", val_col: str = "load",
                    freq: str | None = None) -> tuple[pd.Series, dict, pd.DataFrame]:
    """One-call convenience wrapper used by both the notebook and the app.

    Returns ``(series, grid, quality_report)`` where ``grid`` is the dict
    from :func:`time_grid` and ``series`` is the cleaned load (kW), indexed
    by a regular DatetimeIndex.
    """
    gran = pd.Timedelta(freq) if freq else infer_granularity(raw, time_col)
    grid = time_grid(gran)
    report = check_data(raw, time_col, val_col, gran, grid["dt_hours"], grid["steps_per_day"])
    clean = clean_data(raw, time_col, val_col, gran, grid["dt_hours"], grid["step_min"])
    return clean[val_col], grid, report
