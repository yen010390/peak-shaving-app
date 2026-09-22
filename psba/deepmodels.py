"""Optional classifiers for Table 5: LSTM, FCN, ResNet (need tensorflow) and
KNN-DTW (needs tslearn). All are imported lazily and skipped gracefully by
the pipeline if their library is not installed -- they are not required for
the rest of the app to work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:  # pragma: no cover
    tf = None

try:
    import tslearn.neighbors as tsl
except ImportError:  # pragma: no cover
    tsl = None

LOOK_BACK = 24
DL_TRAIN_CAP = 6000   # cap the (balanced) training set for speed in an interactive app
DL_EPOCHS = 5
DL_HIDDEN = 32
KNN_TRAIN_CAP = 1500
KNN_TEST_CAP = 300


def make_windows(X: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """(N, LOOK_BACK, 5) windows: the last 24 lags (chronological, standardized)
    as one channel, plus slot/dow/is_weekend/month as 4 constant channels
    (broadcast across the window) -- a light-weight stand-in for a full
    time-varying calendar encoding.
    """
    lag_cols = [f"lag_{k}" for k in range(LOOK_BACK, 0, -1)]
    seq = X[lag_cols].values.astype(float)
    mu, sd = float(seq.mean()), float(seq.std() + 1e-6)
    seq_n = (seq - mu) / sd
    slot_max = max(float(X["slot"].max()), 1.0)
    cal = np.stack([
        X["slot"].values / slot_max, X["dow"].values / 6.0,
        X["is_weekend"].values.astype(float), X["month"].values / 12.0,
    ], axis=1)
    cal_rep = np.repeat(cal[:, None, :], LOOK_BACK, axis=1)
    channels = np.concatenate([seq_n[..., None], cal_rep], axis=2)
    return channels, mu, sd


def _backbone(kind: str, inp):
    if kind == "LSTM":
        return layers.LSTM(DL_HIDDEN)(inp)
    x = inp
    if kind == "FCN":
        for f, k in [(32, 8), (64, 5), (32, 3)]:
            x = layers.Activation("relu")(layers.BatchNormalization()(layers.Conv1D(f, k, padding="same")(x)))
    else:  # ResNet: small residual stack
        for f in (16, 32, 32):
            shortcut = layers.BatchNormalization()(layers.Conv1D(f, 1, padding="same")(x))
            y = x
            for k, act in [(8, True), (5, True), (3, False)]:
                y = layers.BatchNormalization()(layers.Conv1D(f, k, padding="same")(y))
                y = layers.Activation("relu")(y) if act else y
            x = layers.Activation("relu")(layers.Add()([shortcut, y]))
    return layers.GlobalAveragePooling1D()(x)


def fit_predict_dl(kind: str, Wtr: np.ndarray, ytr_cls: np.ndarray, Wte: np.ndarray, n_classes: int,
                    seed: int = 42) -> np.ndarray:
    """Train an LSTM/FCN/ResNet classifier and return its 1-step class predictions (1..n_classes) on Wte."""
    if tf is None:
        raise ImportError("tensorflow is not installed")
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    ii = rng.permutation(len(Wtr))[:DL_TRAIN_CAP] if len(Wtr) > DL_TRAIN_CAP else np.arange(len(Wtr))
    inp = keras.Input(Wtr.shape[1:])
    out = layers.Dense(n_classes, activation="softmax")(_backbone(kind, inp))
    model = keras.Model(inp, out)
    model.compile("adam", "categorical_crossentropy")
    y_onehot = keras.utils.to_categorical(ytr_cls[ii] - 1, n_classes)
    model.fit(Wtr[ii], y_onehot, epochs=DL_EPOCHS, batch_size=64, verbose=0)
    return model.predict(Wte, verbose=0).argmax(1) + 1


def fit_predict_knn_dtw(Wtr: np.ndarray, ytr_cls: np.ndarray, Wte: np.ndarray, seed: int = 42):
    """Train a KNN-DTW classifier on a capped train subset and predict on a capped, random test
    subset (DTW is O(train x test), too slow for the full test set in an interactive app).
    Returns (test_positions, predictions) -- test_positions indexes into Wte / the test set.
    """
    if tsl is None:
        raise ImportError("tslearn is not installed")
    rng = np.random.default_rng(seed)
    tr_idx = rng.permutation(len(Wtr))[:min(KNN_TRAIN_CAP, len(Wtr))]
    te_idx = rng.permutation(len(Wte))[:min(KNN_TEST_CAP, len(Wte))]
    clf = tsl.KNeighborsTimeSeriesClassifier(
        n_neighbors=3, metric="dtw", n_jobs=-1,
        metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 3},
    )
    clf.fit(Wtr[tr_idx][:, :, :1], ytr_cls[tr_idx])
    pred = clf.predict(Wte[te_idx][:, :, :1])
    return te_idx, pred
