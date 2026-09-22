# 🔋 Peak-Shaving Battery Analysis

A Streamlit app that reproduces the forecasting-based battery peak-shaving
pipeline of *Rafayal & Cevik, "Time series forecasting-based peak shaving
for building energy management," CASCON'22* — generalised to run on any
load (kW) time-series CSV, with the battery, thresholds, and evaluation
window derived from **your** data rather than hard-coded to the paper's.

Pipeline: data validation & cleaning → percentile-based load classes
(C1–C7), thresholds chosen by grid search → regression forecasting
(Naive / RF / LightGBM / XGBoost) and classification (RF / LightGBM /
XGBoost / KNN-DTW / LSTM / FCN / ResNet) → the model Table 4/5 finds most
accurate for *this* dataset drives Algorithm 1 (forecast-driven
charge/discharge) → cost tables under time-of-use + peak-demand billing,
with rate sensitivity.

The app is organised into four tabs, with every table and figure named
and numbered (Table 1–7, Figure 1–7):
1. **EDA & data characteristics** — data-quality summary, dataset
   characteristics (Table 1), class counts (Table 2), ten days of load
   and its distribution (Figure 1), a sample day's load classes (Figure 2),
   and weekday-vs-weekend peak load (Figure 3).
2. **Experimental setup** — default/fixed parameters (Table 3a),
   hyperparameters and how each was chosen (Table 3b), and the charge/
   discharge threshold percentile search (Table 3c).
3. **Forecasting model performance** — regression (Table 4, broken down
   by load level) and classification (Table 5) model comparisons, a
   sample-day forecast (Figure 4), and a confusion matrix for the best
   classifier (Figure 5). The best model of each kind is shown above the
   tabs.
4. **Peak shaving results** — peak-shaving performance (Table 6,
   highlighting the lowest-cost policy), normal vs. high-capacity battery
   (Table 6b), rate sensitivity (Table 7), before/after and battery-SOC
   plots (Figure 6), the three prediction horizons compared (Figure 7),
   an Excel export, and a closing conclusion on how much the battery
   charged/discharged and saved.

```
peak-shaving-app/
├── app.py                    # Streamlit UI
├── psba/                     # the pipeline, as a plain importable package
│   ├── data.py                #   validation, cleaning, granularity detection
│   ├── features.py            #   lags, calendar features, C1-C7 classes
│   ├── battery.py             #   battery sizing from P_rate (kW) and capacity (kWh)
│   ├── algorithm.py           #   Algorithm 1 + billing cost model
│   ├── models.py              #   RF / LightGBM / XGBoost + metrics
│   ├── deepmodels.py          #   optional: LSTM / FCN / ResNet / KNN-DTW
│   └── pipeline.py            #   ties it all together -> Result dataclass
├── requirements.txt
├── colab/run_in_colab.ipynb  # one-click Colab launcher (Streamlit + localtunnel)
└── 06_household_electricity.csv   # bundled sample dataset (optional)
```

`deepmodels.py`'s models need `tensorflow` and `tslearn` (see
`requirements.txt`) — heavy, optional dependencies. If they are not
installed, selecting KNN-DTW/LSTM/FCN/ResNet in the sidebar just skips
them; every other model and tab works without them.

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/peak-shaving-app.git
cd peak-shaving-app
pip install -r requirements.txt
streamlit run app.py
```

Open the printed `http://localhost:8501` link, upload your CSV (or tick
"use the bundled sample file"), pick options in the sidebar, and click
**Run pipeline**.

## Run in Google Colab

Colab does not expose ports directly, so the app is launched inside Colab
and exposed through a public tunnel.

1. Open `colab/run_in_colab.ipynb` in Colab
   (**File → Open notebook → GitHub**, paste this repo's URL), or click:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/peak-shaving-app/blob/main/colab/run_in_colab.ipynb)

2. Set `REPO_URL` in cell 1 to your fork if you changed anything.
3. **Runtime → Run all.**
4. Open the `https://*.loca.lt` URL printed by the last cell. If it asks
   for a "Tunnel Password", use the IP printed by the cell just before it.

No ngrok account/token is required (it uses `localtunnel`, via `npx`/`npm`,
which Colab already has installed). If `loca.lt` is blocked on your network,
swap in `pyngrok` instead — see the comment at the top of the last cell.

## Data format

A CSV with:
- a timestamp column named `datetime` at a **regular** interval (15 min,
  30 min, 1 h, …) — the app auto-detects the granularity from the
  timestamps (`mode(t[i+1] - t[i])`), no need to tell it the interval;
- a load column named `load`, in **kW**.

Rename your columns to match, or edit `TIME_COL`/`VAL_COL` near the top
of `app.py`.

Missing steps are handled automatically, not a toggle: short gaps (<=1h)
are time-interpolated, longer gaps are filled from the seasonal median
for the same weekday + time-of-day, and any remainder is
forward/back-filled.

## What's configurable vs. fixed

Configurable in the sidebar: battery power rating and capacity (default
300 kW / 1,200 kWh, the paper's), whether to also evaluate a
high-capacity (2x) battery, electricity rates (off-peak / on-peak /
peak-charge), tree-count search, which regression/classification models
to compare, test-set and evaluation-window length, "fast mode" for a
quicker first run.

Always automatic (not exposed as a toggle, to keep the sidebar focused):
the charge/discharge threshold percentile pair (grid search on a
held-out slice — see Table 3c), the SOC convention (the physically
consistent `delta (kWh) = P_rate x dt` formula), and on-peak hours
(widest-window search for the highest mean load). Table 3b in the
Experimental setup tab always shows the method used for each.

Not yet exposed in the UI (edit `psba/pipeline.Config` or the source if
you need them): the Markov-Decision-Process variant explored alongside
this app, and the class-balancing strategy (fixed to oversampling every
class to the largest, `Config.resample_target`).

## License / attribution

This is an independent, from-scratch re-implementation for analysis
purposes; it is not the paper's original code. Cite the paper if you use
this for research:

> S. Rafayal and M. Cevik, "Time series forecasting-based peak shaving for
> building energy management," CASCON'22.
"# peak-shaving-app" 
