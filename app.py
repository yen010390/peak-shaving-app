"""Streamlit UI for the peak-shaving pipeline (psba package).

Run locally:      streamlit run app.py
Run in Colab:      see README.md / colab/run_in_colab.ipynb
"""
from __future__ import annotations

import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from psba import algorithm as alg
from psba.pipeline import Config, run_pipeline

st.set_page_config(page_title="Peak-Shaving Battery Analysis", layout="wide")
st.title("Forecast-based peak-shaving analysis")
st.caption(
    "Reproduces the forecasting + battery peak-shaving pipeline of "
    "*Rafayal & Cevik, \"Time series forecasting-based peak shaving for building "
    "energy management,\" CASCON'22*, generalised to any load (kW) CSV."
)

# CSV column names (fixed; change here if your file uses different ones) ----
TIME_COL = "datetime"
VAL_COL = "load"

# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Load CSV (timestamp + load in kW)", type=["csv"])
    use_sample = st.checkbox("Use the bundled sample file", value=up is None)
    st.caption(f"Expected columns: '{TIME_COL}' (timestamp) and '{VAL_COL}' (kW).")

    st.header("2. Battery")
    st.caption(
        "Defaults (300 kW / 1,200 kWh) are the paper's, sized for a commercial "
        "building averaging ~3,186 kW. If your data's mean load is much "
        "smaller, scale these down too, or the battery's own charging will "
        "create new, bigger peaks."
    )
    battery_p_rate = st.number_input("Charge/discharge power rating, P_rate (kW)", min_value=0.1, value=300.0, step=10.0)
    battery_capacity = st.number_input("Total capacity (kWh)", min_value=0.1, value=1200.0, step=50.0)
    high_capacity = st.checkbox("Also evaluate a high-capacity battery (2x P_rate and capacity)", value=True)

    st.header("3. Electricity rates")
    rate_off = st.number_input("Off-peak rate ($/kW)", min_value=0.0, value=0.14885, step=0.001, format="%.5f")
    rate_on = st.number_input("On-peak rate ($/kW)", min_value=0.0, value=0.17548, step=0.001, format="%.5f")
    rate_peak = st.number_input("Peak-charge rate ($/kW)", min_value=0.0, value=20.68, step=0.1, format="%.2f")

    st.header("4. Models")
    n_trees_mode = st.radio("Number of trees", ["Auto-search", "Fixed"])
    n_trees = 200
    if n_trees_mode == "Fixed":
        n_trees = st.number_input("N_TREES", min_value=10, value=200, step=10)
    reg_models = st.multiselect("Regression models (Table 4)", ["Naive", "RF", "LGBM", "XGBoost"],
                                 default=["Naive", "RF", "LGBM"])
    clf_models = st.multiselect("Classification models (Table 5)",
                                 ["RF", "LGBM", "XGBoost", "KNN-DTW", "LSTM", "FCN", "ResNet"],
                                 default=["RF", "LGBM", "XGBoost"])
    st.caption("KNN-DTW, LSTM, FCN and ResNet need tensorflow/tslearn, run with reduced settings for speed, and are slower.")

    st.header("5. Evaluation window")
    test_days = st.number_input("Test set length (days)", min_value=7, value=31)
    eval_days = st.number_input("Days averaged for Table 6/7 (0 = all test days)", min_value=0, value=15)

    fast = st.checkbox("Fast mode (fewer trees / smaller grids, for a quick first run)", value=True)

    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def build_config() -> Config:
    return Config(
        time_col=TIME_COL, val_col=VAL_COL,
        test_days=int(test_days), eval_days=(None if eval_days == 0 else int(eval_days)),
        # Charge/discharge thresholds are always auto-selected by grid search on a held-out slice (Config's defaults).
        lmin_grid=[40, 50, 60] if fast else [30, 40, 50, 60],
        lmax_grid=[90, 95, 97] if fast else [85, 90, 93, 95, 97, 98, 99],
        n_trees=int(n_trees), auto_n_trees=(n_trees_mode == "Auto-search"),
        ntrees_grid=[10, 25, 50, 100] if fast else [10, 25, 50, 100, 150, 200, 300],
        battery_p_rate=battery_p_rate, battery_capacity_kwh=battery_capacity,
        # battery_units uses Config's default ('energy': delta in kWh = P_rate x dt, the physically consistent formula).
        high_capacity=high_capacity,
        rate_off=rate_off, rate_on=rate_on, rate_peak=rate_peak,
        models=reg_models or ["RF"], clf_models=clf_models or ["RF"],
    )


# ---------------------------------------------------------- figure/table titles --
FIG = {
    1: "Figure 1. A sample data of electricity load over ten days",
    2: "Figure 2. A sample grouping of electricity load",
    3: "Figure 3. Peak of electricity load on weekdays and weekends",
    4: "Figure 4. Electricity load forecast for a sample day",
    5: "Figure 5. Confusion matrix for the best classification model",
    6: "Figure 6. Peak shaving for a sample day (before / after)",
    7: "Figure 7. Peak shaving results using the three different prediction horizons for a sample day",
    8: "Figure 8. Peak shaving performance for the high-capacity battery",
}
TAB = {
    1: "Table 1. Dataset characteristics",
    2: "Table 2. Instances per class, before / after resampling",
    "3a": "Table 3a. Default parameters",
    "3b": "Table 3b. Hyperparameters",
    "3c": "Table 3c. Charge / discharge threshold percentile search results",
    4: "Table 4. Regression models comparison",
    5: "Table 5. Classification models comparison",
    6: "Table 6. Peak-shaving performance",
    "6b": "Table 6b. Normal vs. high-capacity battery",
    7: "Table 7. Energy cost analysis (energy / peak rate sensitivity)",
}


def _sample_day_mask(res):
    return res.index_test.normalize() == res.sample_day


def _hour_axis(ax):
    """Sample-day plots: show only the time of day on the x-axis, not the (repeated) date."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
    ax.set_xlabel("Time of day")


def _daily_peaks(s: pd.Series, steps_per_day: int) -> pd.DataFrame:
    """One row per full calendar day: its peak load, mean load, and Weekday/Weekend label."""
    day = s.index.normalize()
    counts = s.groupby(day).size()
    full_days = counts.index[counts == steps_per_day]
    peak = s.groupby(day).max().loc[full_days]
    mean_load = s.groupby(day).mean().loc[full_days]
    day_type = np.where(peak.index.dayofweek >= 5, "Weekend", "Weekday")
    return pd.DataFrame({"peak_kW": peak, "mean_kW": mean_load, "day_type": day_type})


# -------------------------------------------------------------- rendering --
def render_results(res):
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean load", f"{res.series.mean():,.2f} kW")
    c2.metric("Granularity", f"{res.grid['step_min']} min")
    c3.metric("Charge threshold / Discharge threshold", f"{res.L_MIN:,.1f} / {res.L_MAX:,.1f} kW")

    if res.battery.p_rate_kw > 3 * res.series.mean():
        st.warning(
            f"The battery's power rating ({res.battery.p_rate_kw:,.0f} kW) is more than 3x this "
            f"data's mean load ({res.series.mean():,.1f} kW). Each charge/discharge step then "
            "moves the net load by more than the load itself typically varies, which can create "
            "new, bigger peaks instead of shaving them — watch for negative 'saving' figures "
            "in Table 6 below. Lower P_rate / capacity in the sidebar to match this dataset's scale."
        )

    b1, b2 = st.columns(2)
    b1.success(f"Best regression model: **{res.best_reg}** (lowest ND on the test set, Table 4)")
    b2.success(f"Best classification model: **{res.best_clf}** (highest macro F1-score, Table 5)")

    tab_eda, tab_setup, tab_forecast, tab_peak = st.tabs([
        "EDA & data characteristics", "Experimental setup", "Forecasting model performance", "Peak shaving results",
    ])

    with tab_eda:
        render_eda(res)
    with tab_setup:
        render_setup(res)
    with tab_forecast:
        render_forecasting(res)
    with tab_peak:
        render_peak_shaving(res)


def render_eda(res):
    n_pass = int((res.quality["status"] == "PASS").sum())
    n_total = len(res.quality)
    if n_pass == n_total:
        st.success(f"Data-quality checks: all {n_total} checks passed.")
    else:
        warns = ", ".join(res.quality.index[res.quality["status"] == "WARN"])
        st.warning(f"Data-quality checks: {n_pass}/{n_total} passed. Flagged: {warns}.")

    st.subheader(TAB[1])
    st.dataframe(res.table1, use_container_width=True)

    st.subheader(TAB[2])
    st.dataframe(res.table2, use_container_width=True)

    st.subheader(FIG[1])
    s = res.series
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 3.2))
    w = s.iloc[: res.grid["steps_per_day"] * 10]
    ax1a.plot(w.index, w.values, "k", lw=1)
    ax1a.set(title="(a) Ten days of load", xlabel="Time", ylabel="Load (kW)")
    ax1a.grid(ls=":")
    ax1b.hist(s.values, bins=50, color="tab:blue")
    ax1b.set(title="(b) Load distribution", xlabel="Load (kW)", ylabel="Count")
    ax1b.grid(ls=":", axis="y")
    fig1.autofmt_xdate()
    st.pyplot(fig1)

    st.subheader(FIG[2])
    m = _sample_day_mask(res)
    day_vals = res.actual_test[m]
    day_idx = res.index_test[m]
    edges = res.edges
    charge_max, discharge_min, n_classes = res.forecasts["charge_max"], res.forecasts["discharge_min"], res.forecasts["n_classes"]
    fig2, ax2 = plt.subplots(figsize=(10, 4.2))
    for i in range(1, n_classes + 1):
        lo, hi = edges[i - 1], edges[i]
        if i <= charge_max:
            color, lbl = "#CFE8CB", (f"Charge region (C1-C{charge_max})" if i == 1 else None)
        elif i >= discharge_min:
            color, lbl = "#F1C6B3", (f"Discharge region (C{discharge_min}-C{n_classes})" if i == discharge_min else None)
        else:
            color, lbl = "#E4E1D8", ("No-action region" if i == charge_max + 1 else None)
        ax2.axhspan(lo, hi, color=color, alpha=0.6, label=lbl)
        ax2.text(day_idx.min(), (lo + hi) / 2, f"C{i}", va="center", ha="left", fontsize=9, color="#333333")
    ax2.plot(day_idx, day_vals, "k-o", ms=3, lw=1.2)
    ax2.set(ylabel="Load (kW)")
    _hour_axis(ax2)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(ls=":")
    st.pyplot(fig2)
    st.caption(
        f"Classes computed over the training data (most of the dataset). Charge: load at or below "
        f"{res.L_MIN:,.1f} kW; discharge: load at or above {res.L_MAX:,.1f} kW."
    )

    st.subheader(FIG[3])
    dp = _daily_peaks(res.series, res.grid["steps_per_day"])
    wd = dp.loc[dp.day_type == "Weekday", "peak_kW"]
    we = dp.loc[dp.day_type == "Weekend", "peak_kW"]
    if len(wd) and len(we):
        c1, c2 = st.columns(2)
        c1.metric("Weekday avg. daily peak", f"{wd.mean():,.1f} kW")
        c2.metric("Weekend avg. daily peak", f"{we.mean():,.1f} kW",
                   delta=f"{100 * (we.mean() / wd.mean() - 1):.1f}% vs. weekday")

        fig3, ax3 = plt.subplots(1, 2, figsize=(10, 3.6))
        ax3[0].boxplot([wd, we], labels=["Weekday", "Weekend"], showmeans=True)
        ax3[0].set(title="(a) Daily peak load", ylabel="Peak load (kW)")
        ax3[0].grid(ls=":", axis="y")

        hour_of_day = (res.series.index.hour * 60 + res.series.index.minute) / 60
        day_type_all = np.where(res.series.index.dayofweek >= 5, "Weekend", "Weekday")
        for label, color in [("Weekday", "tab:blue"), ("Weekend", "tab:orange")]:
            mm = day_type_all == label
            profile = res.series[mm].groupby(hour_of_day[mm]).mean()
            ax3[1].plot(profile.index, profile.values, label=label, color=color)
        ax3[1].set(title="(b) Average daily load profile", xlabel="Hour of day", ylabel="Load (kW)", xlim=(0, 24))
        ax3[1].legend(fontsize=8); ax3[1].grid(ls=":")
        st.pyplot(fig3)
    else:
        st.caption("Not enough full weekday and weekend days in this data to compare.")


def render_setup(res):
    st.subheader(TAB["3a"])
    st.dataframe(res.table3_defaults, use_container_width=True)

    st.subheader(TAB["3b"])
    st.dataframe(res.table3, use_container_width=True)

    if res.forecasts["grid_df"] is not None:
        st.subheader(TAB["3c"])
        st.dataframe(res.forecasts["grid_df"].head(10), use_container_width=True)
        st.caption(
            f"Chosen pair: charge threshold at percentile {res.forecasts['lmin']} (= {res.L_MIN:,.1f} kW), "
            f"discharge threshold at percentile {res.forecasts['lmax']} (= {res.L_MAX:,.1f} kW) — lowest "
            "hold-out cost. The percentile and its kW value are the same threshold; percentiles are computed "
            "on the full training set, so they match the 'Charge threshold / Discharge threshold' metric above."
        )


def render_forecasting(res):
    st.subheader(TAB[4])
    st.dataframe(res.table4.round(4).style.highlight_min(axis=0, color="#ffe08a"), use_container_width=True)
    st.caption("Subsets by actual load class: All test data, at/above C5 (no-action ceiling), C6 only, C7 (highest load) only.")

    st.subheader(TAB[5])
    st.dataframe(
        res.table5.style.highlight_max(subset=["Accuracy", "F1-score"], color="#ffe08a"),
        use_container_width=True,
    )

    st.subheader(FIG[4])
    m = _sample_day_mask(res)
    day_idx = res.index_test[m]
    fig4, ax4 = plt.subplots(figsize=(10, 3.4))
    ax4.plot(day_idx, res.actual_test[m], "k", label="Actual", lw=2)
    for name, pred in res.forecasts["reg_preds"].items():
        pred = np.asarray(pred, dtype=float)
        if not np.isnan(pred[m]).all():
            ax4.plot(day_idx, pred[m], label=name, alpha=0.85)
    ax4.set(ylabel="Load (kW)")
    _hour_axis(ax4)
    ax4.legend(); ax4.grid(ls=":")
    st.pyplot(fig4)
    st.caption("Classification predicts classes C1 (lowest load) to C7 (highest load); see Table 5 above.")

    st.subheader(FIG[5])
    y_true = res.forecasts["Cte"]
    pred = np.asarray(res.forecasts["clf_preds"][res.best_clf], dtype=float)
    mask = ~np.isnan(pred)
    n_classes = res.forecasts["n_classes"]
    cm = confusion_matrix(y_true[mask], pred[mask].astype(int), labels=list(range(1, n_classes + 1)))
    fig5, ax5 = plt.subplots(figsize=(5, 4.6))
    ax5.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax5.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if v > cm.max() / 2 else "black")
    ax5.set(xticks=range(n_classes), xticklabels=[f"C{k}" for k in range(1, n_classes + 1)],
            yticks=range(n_classes), yticklabels=[f"C{k}" for k in range(1, n_classes + 1)],
            xlabel="Predicted class", ylabel="True class", title=f"Model: {res.best_clf}")
    st.pyplot(fig5)


def render_peak_shaving(res):
    st.subheader(f"{TAB[6]} ({len(res.eval_dates)} days averaged)")
    best_row = res.table6["Total cost ($/day, avg)"].astype(float).idxmin()
    st.dataframe(
        res.table6.round(2).style.apply(
            lambda col: ["background-color: #ffe08a" if i == best_row else "" for i in col.index],
            subset=["Total cost ($/day, avg)"],
        ),
        use_container_width=True,
    )
    st.caption(f"Lowest total cost: **{best_row}**.")

    if res.table6b is not None:
        st.subheader(TAB["6b"])
        st.dataframe(res.table6b.round(2), use_container_width=True)

    st.subheader(TAB[7])
    st.dataframe(res.table7.round(2), use_container_width=True)

    candidates = [k for k in res.table6.index if k != "No battery" and not k.startswith("Perfect")]
    best_policy_name = res.table6.loc[candidates, "Total cost ($/day, avg)"].astype(float).idxmin()
    A, soc = res.policies[best_policy_name]
    m = _sample_day_mask(res)
    net = alg.apply_actions(res.actual_test, A, res.battery.p_rate_kw)

    st.subheader(FIG[6])
    fig6, ax6 = plt.subplots(1, 2, figsize=(12, 3.5))
    ax6[0].plot(res.index_test[m], res.actual_test[m], "r", label="Before")
    ax6[0].plot(res.index_test[m], net[m], "g", label="After")
    ax6[0].set(title=best_policy_name, ylabel="Net flow (kW)")
    _hour_axis(ax6[0])
    ax6[0].legend(); ax6[0].grid(ls=":")
    ax6[1].fill_between(np.arange(m.sum()), soc[m], color="skyblue")
    ax6[1].set(title="Battery state of charge", xlabel="Step of day", ylabel="SOC (kW-equiv.)")
    ax6[1].grid(ls=":")
    st.pyplot(fig6)

    st.subheader(FIG[7])
    kind_label = "classification" if "classification" in best_policy_name else "regression"
    model_name = best_policy_name.split(" ")[0]
    fig7, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax_h, H in zip(axes, res.config.horizons):
        pname = f"{model_name} {kind_label} - {H} step"
        if pname not in res.policies:
            ax_h.axis("off")
            continue
        A_h, _ = res.policies[pname]
        net_h = alg.apply_actions(res.actual_test, A_h, res.battery.p_rate_kw)
        ax_h.plot(res.index_test[m], res.actual_test[m], "r", label="Before")
        ax_h.plot(res.index_test[m], net_h[m], "g", label="After")
        ax_h.set_title(f"{H}-step horizon")
        _hour_axis(ax_h)
        ax_h.grid(ls=":")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Net flow (kW)")
    fig7.suptitle(f"{model_name}, {kind_label}")
    st.pyplot(fig7)

    if res.table6b is not None:
        st.subheader(FIG[8])
        net_hc = alg.apply_actions(res.actual_test, A, res.battery.p_rate_kw, scale=2.0)
        soc_hc = soc * 2.0  # SOC's unit path is scale-independent; only its kW value doubles
        fig8, ax8 = plt.subplots(1, 2, figsize=(12, 3.5))
        ax8[0].plot(res.index_test[m], res.actual_test[m], "r", label="Before")
        ax8[0].plot(res.index_test[m], net_hc[m], "g", label="After")
        ax8[0].set(title=f"{best_policy_name} ({2 * res.battery.p_rate_kw:,.0f} kW / {2 * res.battery.capacity_kwh:,.0f} kWh)",
                    ylabel="Net flow (kW)")
        _hour_axis(ax8[0])
        ax8[0].legend(); ax8[0].grid(ls=":")
        ax8[1].fill_between(np.arange(m.sum()), soc_hc[m], color="skyblue")
        ax8[1].set(title="Battery state of charge (high-capacity)", xlabel="Step of day", ylabel="SOC (kW-equiv.)")
        ax8[1].grid(ls=":")
        st.pyplot(fig8)

    st.download_button(
        "Download all tables as Excel",
        data=_tables_to_excel(res),
        file_name="peak_shaving_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Conclusion: charging and discharging")
    row = res.table6.loc[best_policy_name]
    n_shaved, n_peaks = row["# peaks shaved"].split(" / ")
    st.error(
        f"The best forecast-driven policy, **{best_policy_name}**, charged the battery "
        f"**{int(row['# charges'])} times** and discharged it **{int(row['# discharges'])} times** "
        f"over the {len(res.eval_dates)}-day evaluation window. It shaved **{n_shaved} of {n_peaks}** "
        f"peak-load days and cut the total electricity cost by **{row['Saving vs no battery (%)']:.2f}%** "
        f"(${res.table6.loc['No battery', 'Total cost ($/day, avg)'] - row['Total cost ($/day, avg)']:.2f}/day) "
        "compared to no battery. Charging happens when the forecast load drops into the charge region "
        "(C1-C2, at or below the charge threshold); discharging happens when it rises into the discharge "
        "region (C6-C7, at or above the discharge threshold) and the battery has enough charge left for "
        "any bigger peak forecast ahead."
    )


def _tables_to_excel(res) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        res.table1.to_excel(writer, sheet_name="Table 1")
        res.table2.to_excel(writer, sheet_name="Table 2")
        res.table3_defaults.to_excel(writer, sheet_name="Table 3a")
        res.table3.to_excel(writer, sheet_name="Table 3b")
        res.table4.to_excel(writer, sheet_name="Table 4")
        res.table5.to_excel(writer, sheet_name="Table 5")
        res.table6.to_excel(writer, sheet_name="Table 6")
        if res.table6b is not None:
            res.table6b.to_excel(writer, sheet_name="Table 6b")
        res.table7.to_excel(writer, sheet_name="Table 7")
    return buf.getvalue()


# ------------------------------------------------------------------- main --
if run_btn:
    if up is not None:
        raw = _read_csv(up.getvalue())
    elif use_sample:
        raw = pd.read_csv("06_household_electricity.csv")
    else:
        st.warning("Upload a CSV or tick 'use the bundled sample file'.")
        st.stop()

    cfg = build_config()
    prog = st.progress(0.0, text="Starting...")
    try:
        res = run_pipeline(raw, cfg, progress=lambda f, m: prog.progress(f, text=m))
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()
    prog.empty()
    st.success("Done.")
    st.session_state["result"] = res

if "result" in st.session_state:
    render_results(st.session_state["result"])
else:
    st.info("Configure the run in the sidebar, then click **Run pipeline**.")
