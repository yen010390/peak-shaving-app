"""End-to-end pipeline: data -> features -> classes -> models -> Algorithm 1 -> tables.

This module has no Streamlit dependency, so it can be unit-tested and reused
from a plain script or a notebook; ``app.py`` only handles the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor

from . import algorithm as alg
from . import battery as bat
from . import data as dat
from . import deepmodels as dm
from . import features as feat
from . import models as mdl

HORIZONS_DEFAULT = [1, 3, 6]


@dataclass
class Config:
    time_col: str = "datetime"
    val_col: str = "load"
    freq: str | None = None                 # None -> auto-detect

    test_days: int = 31
    val_days: int = 45

    lmin: int = 50
    lmax: int = 90
    auto_select_percentiles: bool = True
    lmin_grid: list[int] = field(default_factory=lambda: [30, 40, 50, 60])
    lmax_grid: list[int] = field(default_factory=lambda: [85, 90, 93, 95, 97, 98, 99])
    n_charge: int = 2
    n_mid: int = 3
    n_discharge: int = 2

    resample_target: str | None = "max"      # fixed class-balancing for the classifier's training set: 'max' | 'mean' | None

    n_trees: int = 200
    auto_n_trees: bool = True
    ntrees_grid: list[int] = field(default_factory=lambda: [10, 25, 50, 100, 150, 200, 300])
    ntrees_tol: float = 0.005

    battery_p_rate: float = bat.PAPER_P_RATE_KW           # kW, charge/discharge power rating
    battery_capacity_kwh: float = bat.PAPER_CAPACITY_KWH  # kWh, total capacity
    battery_units: str = "energy"          # delta in kWh = P_rate x dt (the physically consistent formula); 'paper' reproduces Fig. 5b of the paper instead
    high_capacity: bool = True                            # also evaluate a 2x P_rate / 2x capacity battery

    rate_off: float = 0.14885
    rate_on: float = 0.17548
    rate_peak: float = 20.68
    auto_on_peak: bool = True
    on_peak_width: int = 5
    on_peak_hours: list[int] = field(default_factory=lambda: list(range(16, 21)))

    horizons: list[int] = field(default_factory=lambda: list(HORIZONS_DEFAULT))
    models: list[str] = field(default_factory=lambda: ["Naive", "RF", "LGBM"])
    clf_models: list[str] = field(default_factory=lambda: ["RF", "LGBM", "XGBoost"])

    eval_days: int | None = 15
    eval_window: str = "last"                 # 'last' | 'first'

    seed: int = 42


@dataclass
class Result:
    series: pd.Series
    grid: dict
    quality: pd.DataFrame
    config: Config
    L_MIN: float
    L_MAX: float
    edges: np.ndarray
    battery: bat.Battery
    table1: pd.DataFrame
    table2: pd.DataFrame
    table3: pd.DataFrame           # hyperparameters, with the method used to set each
    table3_defaults: pd.DataFrame  # fixed / default parameters (battery, rates, split, seed)
    table4: pd.DataFrame
    table5: pd.DataFrame
    table6: pd.DataFrame
    table6b: pd.DataFrame | None       # normal vs. high-capacity battery, best policy (None if not requested)
    table7: pd.DataFrame
    forecasts: dict
    policies: dict
    index_test: pd.DatetimeIndex
    actual_test: np.ndarray
    eval_dates: pd.DatetimeIndex
    sample_day: pd.Timestamp                # day (within eval_dates) with the highest actual peak; used by every figure
    best_reg: str                           # most accurate regression model (Table 4)
    best_clf: str                           # most accurate classification model (Table 5)
    algo1_reg_model: str                    # model actually driving Algorithm 1's regression policy (Table 6/7)
    algo1_clf_model: str                    # model actually driving Algorithm 1's classification policy (Table 6/7)


def _best_window(hour_profile: pd.Series, width: int):
    means = {h: np.mean([hour_profile[(h + k) % 24] for k in range(width)]) for h in range(24)}
    h0 = max(means, key=means.get)
    return [(h0 + k) % 24 for k in range(width)], means[h0]


def run_pipeline(raw: pd.DataFrame, cfg: Config, progress=None) -> Result:
    """Run the whole pipeline. ``progress(fraction, message)`` is called if given."""

    def step(frac, msg):
        if progress is not None:
            progress(frac, msg)

    # 1. data --------------------------------------------------------------
    step(0.02, "Validating and cleaning data...")
    s, grid, quality = dat.load_and_clean(raw, cfg.time_col, cfg.val_col, cfg.freq)
    mean_load = s.mean()
    steps_per_day, dt_hours, step_min, lags = grid["steps_per_day"], grid["dt_hours"], grid["step_min"], grid["lags"]

    test_start = s.index.max().normalize() - pd.Timedelta(days=cfg.test_days - 1)
    s_tr = s[s.index < test_start]

    # battery, sized directly from P_rate (kW) and capacity (kWh) -------------
    battery = bat.size_battery(cfg.battery_p_rate, cfg.battery_capacity_kwh, mean_load, dt_hours, cfg.battery_units)

    # 2. on-peak hours -------------------------------------------------------
    if cfg.auto_on_peak:
        hour_profile = s_tr.groupby(s_tr.index.hour).mean()
        on_peak_hours = _best_window(hour_profile, cfg.on_peak_width)[0]
    else:
        on_peak_hours = list(cfg.on_peak_hours)

    # 3. features & split -----------------------------------------------------
    step(0.08, "Building features...")
    X = feat.make_features(s, lags, step_min)
    h_max = max(cfg.horizons)
    Y = feat.make_targets(s, h_max)
    ok = ~np.isnan(X.values).any(axis=1) & ~np.isnan(Y).any(axis=1)
    X, Y, index = X[ok], Y[ok], X.index[ok]
    is_tr = index < test_start
    Xtr, Ytr = X[is_tr], Y[is_tr]
    Xte, Yte = X[~is_tr], Y[~is_tr]
    index_test = index[~is_tr]
    actual_test = Yte[:, 0]

    val_start = test_start - pd.Timedelta(days=cfg.val_days)
    is_fit = Xtr.index < val_start
    Xfit, Yfit = Xtr[is_fit], Ytr[is_fit]
    Xva, Yva = Xtr[~is_fit], Ytr[~is_fit]
    colsample = np.sqrt(X.shape[1]) / X.shape[1]

    # 4. N_TREES search (simplified: one time-series split) -------------------
    n_trees = cfg.n_trees
    ntrees_curve = None
    if cfg.auto_n_trees:
        step(0.15, "Searching the number of trees...")
        kmax = max(cfg.ntrees_grid)
        splits = list(TimeSeriesSplit(3).split(Xtr))
        maes = {k: [] for k in cfg.ntrees_grid}
        for a, b in splits:
            rf = mdl.make_rf("reg", kmax, colsample, cfg.seed).fit(Xtr.iloc[a], Ytr[a, 0])
            preds = np.stack([t.predict(Xtr.iloc[b].values) for t in rf.estimators_])
            for k in cfg.ntrees_grid:
                maes[k].append(np.abs(preds[:k].mean(0) - Ytr[b, 0]).mean())
        curve = pd.DataFrame({"n_trees": cfg.ntrees_grid, "cv_mae": [np.mean(maes[k]) for k in cfg.ntrees_grid]})
        curve["vs_best_%"] = 100 * (curve.cv_mae / curve.cv_mae.min() - 1)
        n_trees = int(curve.loc[curve["vs_best_%"] <= 100 * cfg.ntrees_tol, "n_trees"].min())
        ntrees_curve = curve

    # 5. percentile selection (simplified: single hold-out) -------------------
    step(0.25, "Selecting the charge and discharge thresholds...")
    grid_df = None
    lmin, lmax = cfg.lmin, cfg.lmax
    if cfg.auto_select_percentiles:
        va_actual = Yva[:, 0]
        base_cost = alg.daily_cost(va_actual, Xva.index, on_peak_hours, cfg.rate_off, cfg.rate_on,
                                    cfg.rate_peak, dt_hours, steps_per_day)[0].sum()
        rf_va1 = mdl.make_rf("reg", n_trees, colsample, cfg.seed).fit(Xfit, Yfit[:, 0])
        rf_va2 = mdl.make_rf("reg", n_trees, colsample, cfg.seed).fit(Xfit, Yfit[:, 1:])
        y1v, uv = rf_va1.predict(Xva), rf_va2.predict(Xva)
        rows = []
        for pl in cfg.lmin_grid:
            for ph in cfg.lmax_grid:
                thr = tuple(np.percentile(Yfit[:, 0], [pl, ph]))
                A, _ = alg.peak_shaving(y1v, uv[:, :6], "reg", thr, battery.p_rate_kw, battery.soc_max, battery.soc_min)
                net = alg.apply_actions(va_actual, A, battery.p_rate_kw)
                cost = alg.daily_cost(net, Xva.index, on_peak_hours, cfg.rate_off, cfg.rate_on,
                                       cfg.rate_peak, dt_hours, steps_per_day)[0].sum()
                rows.append(dict(lmin=pl, lmax=ph, cost=cost, saving=100 * (1 - cost / base_cost)))
        grid_df = pd.DataFrame(rows).sort_values("cost")
        lmin, lmax = int(grid_df.iloc[0].lmin), int(grid_df.iloc[0].lmax)

    # 6. final classes & battery ----------------------------------------------
    edges = feat.class_edges(Ytr[:, 0], lmin / 100, lmax / 100, cfg.n_charge, cfg.n_mid, cfg.n_discharge)
    n_classes = cfg.n_charge + cfg.n_mid + cfg.n_discharge
    charge_max, discharge_min = cfg.n_charge, n_classes - cfg.n_discharge + 1
    L_MIN, L_MAX = edges[charge_max], edges[discharge_min - 1]

    Ctr = feat.to_class(Ytr, edges)
    Cte = feat.to_class(Yte, edges)
    bal = feat.balance(Ctr[:, 0], cfg.resample_target, cfg.seed)

    table2 = pd.DataFrame({
        f"C{i+1}": [int((Ctr[:, 0] == i + 1).sum()), int((Ctr[bal, 0] == i + 1).sum())]
        for i in range(n_classes)
    }, index=["Training data", "After re-sampling"])

    # 7. final RF models --------------------------------------------------
    step(0.4, "Training the final models...")
    f1_reg = mdl.make_rf("reg", n_trees, colsample, cfg.seed).fit(Xtr, Ytr[:, 0])
    f2_reg = mdl.make_rf("reg", n_trees, colsample, cfg.seed).fit(Xtr, Ytr[:, 1:])
    f1_clf = mdl.make_rf("clf", n_trees, colsample, cfg.seed).fit(Xtr.iloc[bal], Ctr[bal, 0])
    f2_clf = mdl.make_rf("clf", n_trees, colsample, cfg.seed).fit(Xtr.iloc[bal], Ctr[bal, 1:])
    y1_reg_rf, U_reg_rf = f1_reg.predict(Xte), f2_reg.predict(Xte)
    y1_clf_rf, U_clf_rf = f1_clf.predict(Xte), f2_clf.predict(Xte)

    # 8. Table 4 (regression), by load-level subset --------------------------
    step(0.55, "Evaluating regression models...")
    reg_preds = {"RF": y1_reg_rf}
    naive_pred = s.shift(lags[-1]).reindex(index_test).values
    factory = {"LGBM": mdl.make_lgbm, "XGBoost": mdl.make_xgb}
    for name in cfg.models:
        if name == "Naive":
            reg_preds["Naive"] = naive_pred
            continue
        if name == "RF":
            continue
        try:
            reg_preds[name] = factory[name]("reg", n_trees, colsample, cfg.seed).fit(Xtr, Ytr[:, 0]).predict(Xte)
        except Exception:
            continue
    ok_n = ~np.isnan(naive_pred)
    c_top, c_last = discharge_min - 1, n_classes           # e.g. C5, C7 for the default 2/3/2 split
    subset_defs = {
        "All test data": np.ones(len(actual_test), bool),
        f">= C{c_top}": Cte[:, 0] >= c_top,
        f"C{discharge_min}": Cte[:, 0] == discharge_min,
        f"C{c_last}": Cte[:, 0] == c_last,
    }
    subset_frames = {}
    for sname, smask in subset_defs.items():
        rows = {}
        for name, pred in reg_preds.items():
            pred = np.asarray(pred, dtype=float)
            m = smask & (ok_n if name == "Naive" else np.ones(len(actual_test), bool))
            rows[name] = mdl.regression_metrics(actual_test[m], pred[m]) if m.sum() else {"ND": np.nan, "NRMSE": np.nan, "MAPE": np.nan}
        subset_frames[sname] = pd.DataFrame(rows).T
    table4 = pd.concat(subset_frames, axis=1)

    # 9. Table 5 (classification) -------------------------------------------
    step(0.65, "Evaluating classification models...")
    clf_preds = {"RF": y1_clf_rf}
    clf_rows = {"RF": mdl.classification_metrics(Cte[:, 0], y1_clf_rf, n_classes)}
    need_windows = any(m in ("LSTM", "FCN", "ResNet", "KNN-DTW") for m in cfg.clf_models)
    Wtr = Wte = None
    if need_windows:
        step(0.68, "Preparing windows for deep-learning / KNN-DTW models...")
        Wtr, _, _ = dm.make_windows(Xtr)
        Wte, _, _ = dm.make_windows(Xte)
    for name in cfg.clf_models:
        if name == "RF":
            continue
        pred = None
        try:
            if name in ("LGBM", "XGBoost"):
                m = factory[name]("clf", n_trees, colsample, cfg.seed)
                pred = (m.fit(Xtr.iloc[bal], Ctr[bal, 0] - 1).predict(Xte) + 1) if name == "XGBoost" \
                    else m.fit(Xtr.iloc[bal], Ctr[bal, 0]).predict(Xte)
            elif name in ("LSTM", "FCN", "ResNet"):
                pred = dm.fit_predict_dl(name, Wtr[bal], Ctr[bal, 0], Wte, n_classes, cfg.seed)
            elif name == "KNN-DTW":
                te_idx, pred_sub = dm.fit_predict_knn_dtw(Wtr[bal], Ctr[bal, 0], Wte, cfg.seed)
                pred = np.full(len(Xte), np.nan)
                pred[te_idx] = pred_sub
            else:
                continue
        except Exception:
            continue
        clf_preds[name] = pred
        mask = ~np.isnan(pred) if name == "KNN-DTW" else np.ones(len(pred), bool)
        clf_rows[name] = mdl.classification_metrics(Cte[mask, 0], pred[mask].astype(int), n_classes)
        if name == "KNN-DTW":
            clf_rows[name]["Evaluated on"] = f"{int(mask.sum())} / {len(mask)} test samples (random subset, for speed)"
    table5 = pd.DataFrame(clf_rows).T

    best_reg = table4[("All test data", "ND")].astype(float).idxmin()
    best_clf = table5["F1-score"].astype(float).idxmax()

    # 10. Multi-step (f2) forecasts for Algorithm 1, from the model Table 4/5 found BEST FOR
    # THIS DATASET -- not hard-coded to RF. Naive has no sensible f2, so it falls back to the
    # next-best trainable model; any model whose multi-step fit fails does the same. ----------
    step(0.75, "Preparing forecasts for Algorithm 1...")

    def multistep_reg(name):
        if name == "RF":
            return y1_reg_rf, U_reg_rf
        m2 = MultiOutputRegressor(factory[name]("reg", n_trees, colsample, cfg.seed))
        return reg_preds[name], m2.fit(Xtr, Ytr[:, 1:]).predict(Xte)

    def pick_reg(exclude=()):
        pool = [m for m in table4.index if m != "Naive" and m not in exclude]
        return min(pool, key=lambda m: table4.loc[m, ("All test data", "ND")]) if pool else "RF"

    algo1_reg_model = best_reg if best_reg != "Naive" else pick_reg()
    try:
        y1_algo1_reg, U_algo1_reg = multistep_reg(algo1_reg_model)
    except Exception:
        algo1_reg_model = pick_reg(exclude={algo1_reg_model})
        y1_algo1_reg, U_algo1_reg = multistep_reg(algo1_reg_model)

    def multistep_clf(name):
        if name == "RF":
            return y1_clf_rf, U_clf_rf
        if name in ("LGBM", "XGBoost"):
            cols = []
            for k in range(h_max):
                yk = Ctr[bal, 1 + k]
                if name == "XGBoost":
                    classes = np.unique(yk)
                    mk = factory[name]("clf", n_trees, colsample, cfg.seed).fit(Xtr.iloc[bal], np.searchsorted(classes, yk))
                    cols.append(classes[mk.predict(Xte).astype(int)])
                else:
                    mk = factory[name]("clf", n_trees, colsample, cfg.seed).fit(Xtr.iloc[bal], yk)
                    cols.append(mk.predict(Xte))
            return clf_preds[name], np.column_stack(cols)
        # LSTM / FCN / ResNet: one classifier per future step, reusing the windows built for Table 5
        cols = [dm.fit_predict_dl(name, Wtr[bal], Ctr[bal, 1 + k], Wte, n_classes, cfg.seed) for k in range(h_max)]
        return clf_preds[name], np.column_stack(cols)

    # KNN-DTW only covers a random subset of the test set (see Table 5), so it cannot drive
    # Algorithm 1 (like Naive for regression); fall back to the next-best full-coverage model.
    clf_pool = [m for m in table5.index if m != "KNN-DTW"]
    algo1_clf_model = best_clf if best_clf in clf_pool else max(clf_pool, key=lambda m: table5.loc[m, "F1-score"])
    try:
        y1_algo1_clf, U_algo1_clf = multistep_clf(algo1_clf_model)
    except Exception:
        pool = [m for m in clf_pool if m != algo1_clf_model]
        algo1_clf_model = max(pool, key=lambda m: table5.loc[m, "F1-score"]) if pool else "RF"
        y1_algo1_clf, U_algo1_clf = multistep_clf(algo1_clf_model)

    # 11. Algorithm 1 on test -> Table 6 -------------------------------------
    step(0.8, "Running Algorithm 1 (peak shaving)...")
    thr_reg, thr_clf = (L_MIN, L_MAX), (charge_max, discharge_min)
    policies = {"No battery": (np.zeros(len(actual_test), int), None)}
    for H in cfg.horizons:
        policies[f"Perfect forecast - {H} step"] = alg.peak_shaving(
            actual_test, Yte[:, 1:H + 1], "reg", thr_reg, battery.p_rate_kw, battery.soc_max, battery.soc_min)
    for H in cfg.horizons:
        policies[f"{algo1_reg_model} regression - {H} step"] = alg.peak_shaving(
            y1_algo1_reg, U_algo1_reg[:, :H], "reg", thr_reg, battery.p_rate_kw, battery.soc_max, battery.soc_min)
    for H in cfg.horizons:
        policies[f"{algo1_clf_model} classification - {H} step"] = alg.peak_shaving(
            y1_algo1_clf, U_algo1_clf[:, :H], "clf", thr_clf, battery.p_rate_kw, battery.soc_max, battery.soc_min)

    full_days = pd.DatetimeIndex(sorted(set(
        index_test[pd.Series(1, index=index_test).groupby(index_test.normalize()).transform("sum").values == steps_per_day].normalize()
    )))
    if cfg.eval_days is None:
        eval_dates = full_days
    else:
        eval_dates = full_days[-cfg.eval_days:] if cfg.eval_window == "last" else full_days[:cfg.eval_days]

    mask_eval = np.isin(index_test.normalize(), eval_dates)
    pool_idx = index_test[mask_eval] if mask_eval.any() else index_test
    pool_val = actual_test[mask_eval] if mask_eval.any() else actual_test
    sample_day = pd.Series(pool_val, index=pool_idx).groupby(pool_idx.normalize()).max().idxmax()

    peak_mask = actual_test >= L_MAX
    rows6 = {}
    for name, (A, _) in policies.items():
        net = alg.apply_actions(actual_test, A, battery.p_rate_kw)
        tot, pk = alg.daily_cost(net, index_test, on_peak_hours, cfg.rate_off, cfg.rate_on, cfg.rate_peak,
                                  dt_hours, steps_per_day, eval_dates=eval_dates)
        m = np.isin(index_test.normalize(), eval_dates)
        rows6[name] = {
            "Total cost ($/day, avg)": tot.mean(), "Total cost stdev": tot.std(),
            "Peak-charge ($/day, avg)": pk.mean(),
            "# charges": int((A[m] == 1).sum()), "# discharges": int((A[m] == -1).sum()),
            "# peaks shaved": f"{int(((A == -1) & peak_mask)[m].sum())} / {int(peak_mask[m].sum())}",
        }
    table6 = pd.DataFrame(rows6).T
    base_mean = table6.loc["No battery", "Total cost ($/day, avg)"]
    table6["Saving vs no battery (%)"] = 100 * (1 - table6["Total cost ($/day, avg)"] / base_mean)

    # 11. Table 7 (rate sensitivity) ------------------------------------------
    step(0.9, "Rate sensitivity (Table 7)...")
    best_policy_name = table6.drop("No battery").loc[
        [i for i in table6.index if not i.startswith("Perfect") and i != "No battery"], "Total cost ($/day, avg)"
    ].astype(float).idxmin()
    A_best = policies[best_policy_name][0]
    net_best = alg.apply_actions(actual_test, A_best, battery.p_rate_kw)
    net_nobat = actual_test

    # 10b. Normal vs. high-capacity battery (2x P_rate and capacity), same policy's actions ---
    table6b = None
    if cfg.high_capacity:
        net_best_hc = alg.apply_actions(actual_test, A_best, battery.p_rate_kw, scale=2.0)
        rows6b = {}
        for label, net in [
            ("No battery", net_nobat),
            (f"Normal ({battery.p_rate_kw:.0f} kW / {battery.capacity_kwh:.0f} kWh)", net_best),
            (f"High-capacity ({2 * battery.p_rate_kw:.0f} kW / {2 * battery.capacity_kwh:.0f} kWh)", net_best_hc),
        ]:
            tot, pk = alg.daily_cost(net, index_test, on_peak_hours, cfg.rate_off, cfg.rate_on, cfg.rate_peak,
                                      dt_hours, steps_per_day, eval_dates=eval_dates)
            rows6b[label] = {"Total cost ($/day, avg)": tot.mean(), "Peak-charge ($/day, avg)": pk.mean()}
        table6b = pd.DataFrame(rows6b).T
        table6b["Saving vs no battery (%)"] = 100 * (
            1 - table6b["Total cost ($/day, avg)"] / table6b.loc["No battery", "Total cost ($/day, avg)"])

    rows7 = []
    for e_label, e_mult in [("-30%", 0.7), ("Normal", 1.0), ("+30%", 1.3)]:
        for p_label, p_mult in [("-30%", 0.7), ("Normal", 1.0), ("+30%", 1.3)]:
            for pname, net in [("No battery", net_nobat), (best_policy_name, net_best)]:
                A = policies[pname][0]
                tot, pk = alg.daily_cost(net, index_test, on_peak_hours, cfg.rate_off, cfg.rate_on, cfg.rate_peak,
                                          dt_hours, steps_per_day, e_mult, p_mult, eval_dates)
                m = np.isin(index_test.normalize(), eval_dates)
                rows7.append({
                    "Energy rate": e_label, "Peak rate": p_label, "Policy": pname,
                    "Total cost ($/day, avg)": tot.mean(), "Peak-charge ($/day, avg)": pk.mean(),
                    "# charges": int((A[m] == 1).sum()), "# discharges": int((A[m] == -1).sum()),
                    "# peaks shaved": int(((A == -1) & peak_mask)[m].sum()),
                })
    table7 = pd.DataFrame(rows7)

    resample_desc = {"max": "Oversample every class to the largest class's size",
                      "mean": "Oversample/undersample every class to the average class size"}.get(cfg.resample_target, "No resampling (natural class distribution)")
    table3 = pd.DataFrame({
        "Value": {
            "Regression models compared": ", ".join(cfg.models),
            "Classification models compared": ", ".join(cfg.clf_models),
            "Number of trees (N_TREES)": str(n_trees),
            "Class-balancing strategy": cfg.resample_target or "none",
            "Charge / discharge threshold percentile pair": f"P{lmin} / P{lmax}",
            "On-peak hours": ", ".join(f"{h:02d}h" for h in on_peak_hours),
            "Model driving Algorithm 1 (regression)": algo1_reg_model + (" (fallback from Naive)" if best_reg == "Naive" else ""),
            "Model driving Algorithm 1 (classification)": algo1_clf_model + (" (fallback from KNN-DTW)" if best_clf == "KNN-DTW" else ""),
        },
        "Method": {
            "Regression models compared": "User-selected in the sidebar",
            "Classification models compared": "User-selected in the sidebar",
            "Number of trees (N_TREES)": "3-fold time-series cross-validation; lowest tree count within 0.5% of the best CV MAE" if cfg.auto_n_trees else "User-specified (fixed)",
            "Class-balancing strategy": resample_desc,
            "Charge / discharge threshold percentile pair": "Grid search on a held-out slice of the training data, minimizing cost" if cfg.auto_select_percentiles else "User-specified (fixed)",
            "On-peak hours": "Widest-window search for the highest mean load in the training data" if cfg.auto_on_peak else "User-specified (fixed)",
            "Model driving Algorithm 1 (regression)": "Lowest ND on the test set (Table 4, All test data)",
            "Model driving Algorithm 1 (classification)": "Highest macro F1-score on the test set (Table 5)",
        },
    })
    table3_defaults = pd.DataFrame({"Value": {
        "Battery: power rating (P_rate)": f"{battery.p_rate_kw:,.2f} kW",
        "Battery: capacity": f"{battery.capacity_kwh:,.1f} kWh ({battery.battery_hours:g} h of storage)",
        "Battery: SOC convention": f"{cfg.battery_units} ({battery.steps_full:g} discharges when full)",
        "High-capacity scenario evaluated": "Yes (2x)" if cfg.high_capacity else "No",
        "Electricity rates (off-peak / on-peak / peak-charge)": f"${cfg.rate_off:.5f} / ${cfg.rate_on:.5f} / ${cfg.rate_peak:.2f} per kW",
        "Test set length": f"{cfg.test_days} days",
        "Evaluation window (Table 6/7)": f"{len(eval_dates)} days ({cfg.eval_window})",
        "Random seed": cfg.seed,
    }})

    table1 = pd.DataFrame({"Value": {
        "Date range": f"{s.index.min():%Y-%m-%d} to {s.index.max():%Y-%m-%d}",
        "Length": f"{len(s):,}",
        "Granularity": f"{grid['step_min']} min" if grid["step_min"] < 60 else f"{grid['dt_hours']:g} h",
        "Mean load (kW)": f"{mean_load:,.2f}",
        "Any missing data": "No" if quality.loc["missing / empty steps", "count"] == 0 else "Yes",
    }})

    step(1.0, "Done.")
    return Result(
        series=s, grid=grid, quality=quality,
        config=cfg, L_MIN=L_MIN, L_MAX=L_MAX, edges=edges,
        battery=battery, table1=table1, table2=table2, table3=table3, table3_defaults=table3_defaults, table4=table4, table5=table5,
        table6=table6, table6b=table6b, table7=table7,
        forecasts=dict(reg_preds=reg_preds, clf_preds=clf_preds, y1_reg_rf=y1_reg_rf, U_reg_rf=U_reg_rf,
                        y1_clf_rf=y1_clf_rf, U_clf_rf=U_clf_rf, grid_df=grid_df, ntrees_curve=ntrees_curve,
                        n_trees=n_trees, lmin=lmin, lmax=lmax, on_peak_hours=on_peak_hours,
                        charge_max=charge_max, discharge_min=discharge_min, n_classes=n_classes, Cte=Cte[:, 0]),
        policies=policies, index_test=index_test, actual_test=actual_test, eval_dates=eval_dates,
        sample_day=sample_day, best_reg=best_reg, best_clf=best_clf,
        algo1_reg_model=algo1_reg_model, algo1_clf_model=algo1_clf_model,
    )
