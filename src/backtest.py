"""Walk-forward backtest harness built on sktime.

The fold schedule is produced by sktime's ``ExpandingWindowSplitter``. The
default protocol (180-day initial window, 30-day step, 30-day horizon -> 6
folds) matches the original 1-year benchmark; longer datasets can use larger
windows (e.g. initial 730 / step 90 / horizon 90 for multi-year data). Each
fold refits every model on data up to the cutoff and evaluates the horizon
ahead per series. Per-series models are fit per series; pooled ML models are
fit once on the whole training panel and then predict each series recursively.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.load_data import add_series_key, series_key
from src.models import (
    PER_SERIES_MODELS,
    POOLED_MODELS,
    predict_with_lags,
)

try:  # sktime >= 0.30
    from sktime.split import ExpandingWindowSplitter
except ImportError:  # older sktime
    from sktime.forecasting.model_selection import ExpandingWindowSplitter

INITIAL_WINDOW = 180
STEP_LENGTH = 30
HORIZON = 30


def make_folds(n_days: int, initial_window: int = INITIAL_WINDOW,
               step_length: int = STEP_LENGTH, horizon: int = HORIZON) -> list[tuple[int, int]]:
    """Return list of (cutoff_index, test_start_index, test_end_index) from sktime."""
    splitter = ExpandingWindowSplitter(initial_window=initial_window, step_length=step_length, fh=np.arange(1, horizon + 1))
    folds = []
    for train_idx, test_idx in splitter.split(np.arange(n_days)):
        cutoff = int(train_idx[-1])
        folds.append((cutoff, int(test_idx[0]), int(test_idx[-1])))
    return folds


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE (symmetrized MAPE), the Kaggle competition metric."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(np.abs(y_true - y_pred) / denom) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass
class FoldResult:
    model: str
    fold: int
    series: str
    cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    smape: float
    mae: float
    rmse: float
    mase: float
    n_obs: int
    fit_time_s: float = 0.0


def _mase_denominator(train_series: pd.Series, period: int = 7) -> float:
    """Seasonal-naive in-sample MAE used as the MASE scaling (period=7)."""
    vals = train_series.values
    if len(vals) <= period:
        return float(np.mean(np.abs(np.diff(vals))) + 1e-9) if len(vals) > 1 else 1.0
    err = np.abs(vals[period:] - vals[:-period])
    return float(np.mean(err)) + 1e-9


def _eval_series(s: str, hist: pd.DataFrame, model_names: list[str], pooled_fits: dict,
                 test_dates: pd.DatetimeIndex, cutoff: pd.Timestamp, horizon: int,
                 yearly_lags: bool) -> tuple[list[FoldResult], dict]:
    """Evaluate all models for one series at one fold (runs in a worker when n_jobs>1)."""
    results: list[FoldResult] = []
    preds_out: dict[str, pd.Series] = {}
    y_true = hist.set_index("date")["sales"].reindex(test_dates).values.astype(float)
    if np.isnan(y_true).any():
        return results, preds_out

    for mname in model_names:
        t0_ = time.time()
        if mname in PER_SERIES_MODELS:
            history = hist[hist["date"] <= cutoff].set_index("date")["sales"]
            try:
                fc = PER_SERIES_MODELS[mname](history, horizon)
            except Exception:
                fc = None
        elif mname in POOLED_MODELS:
            fc = predict_with_lags(pooled_fits[mname], hist, cutoff, horizon, yearly_lags=yearly_lags)
        else:
            continue

        if fc is None or len(fc) != len(test_dates):
            preds = np.full(len(test_dates), np.nan)
        else:
            fc = fc.reindex(test_dates)
            preds = fc.values.astype(float)
        preds_out[mname] = pd.Series(preds, index=test_dates, name="pred")

        mask = ~np.isnan(y_true) & ~np.isnan(preds)
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], np.clip(preds[mask], 0, None)
        train_hist = hist[hist["date"] <= cutoff]["sales"]
        results.append(FoldResult(
            model=mname, fold=-1, series=s, cutoff=cutoff,
            test_start=test_dates[0], test_end=test_dates[-1],
            smape=smape(yt, yp), mae=mae(yt, yp), rmse=rmse(yt, yp),
            mase=mae(yt, yp) / _mase_denominator(train_hist),
            n_obs=int(mask.sum()), fit_time_s=time.time() - t0_,
        ))
    return results, preds_out


def run_backtest(
    df: pd.DataFrame,
    sample_series: list[tuple[int, int]],
    models: list[str] | None = None,
    extra_pooled: bool = True,
    verbose: bool = True,
    initial_window: int = INITIAL_WINDOW,
    step_length: int = STEP_LENGTH,
    horizon: int = HORIZON,
    n_jobs: int = 1,
    yearly_lags: bool | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the walk-forward backtest on the given series.

    Returns (metrics_frame, predictions) where predictions maps
    (model, fold) -> dict[series, pd.Series].
    """
    d = add_series_key(df)
    dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    folds = make_folds(len(dates), initial_window=initial_window, step_length=step_length, horizon=horizon)

    # normalize series identifiers to string keys ("store_item")
    sample_series = [s if isinstance(s, str) else series_key(*s) for s in sample_series]

    # yearly lag features are usable once the panel spans roughly two years
    if yearly_lags is None:
        yearly_lags = (df["date"].max() - df["date"].min()).days >= 700

    per_series = {s: g.sort_values("date") for s, g in d.groupby("series")}

    model_names = models or list(PER_SERIES_MODELS.keys())
    if extra_pooled:
        model_names = list(dict.fromkeys(model_names + list(POOLED_MODELS.keys())))

    results: list[FoldResult] = []
    predictions: dict[tuple[str, int], dict[str, pd.Series]] = {}

    for fold, (cutoff_idx, t0, t1) in enumerate(folds):
        cutoff = dates[cutoff_idx]
        test_dates = dates[t0:t1 + 1]
        train_panel = d[d["date"] <= cutoff]

        if verbose:
            print(f"\n[fold {fold}] cutoff={cutoff.date()}  test={test_dates[0].date()}..{test_dates[-1].date()}  "
                  f"train_rows={len(train_panel)}")

        # ---- pooled models: fit once per fold ----
        pooled_fits: dict[str, object] = {}
        for mname in POOLED_MODELS:
            if mname not in model_names:
                continue
            t0_ = time.time()
            fit_fn = POOLED_MODELS[mname]
            train_sub = train_panel[train_panel["series"].isin(sample_series)]
            pooled_fits[mname] = fit_fn(train_sub, yearly_lags=yearly_lags)
            if verbose:
                print(f"  fit {mname}: {time.time() - t0_:.1f}s")

        # ---- evaluate each series (parallel across series when n_jobs > 1) ----
        jobs = [
            delayed(_eval_series)(s, per_series[s], model_names, pooled_fits, test_dates, cutoff, horizon, yearly_lags)
            for s in sample_series
        ]
        outcomes = Parallel(n_jobs=max(1, n_jobs), prefer="processes")(jobs)

        for s, (res_list, preds_dict) in zip(sample_series, outcomes):
            for r in res_list:
                r.fold = fold
                results.append(r)
            for mname, p in preds_dict.items():
                predictions.setdefault((mname, fold), {})[s] = p

    metrics = pd.DataFrame([r.__dict__ for r in results])
    return metrics, predictions


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pool metrics across folds/series per model."""
    g = metrics.groupby("model").agg(
        smape=("smape", "mean"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        mase=("mase", "mean"),
        n_obs=("n_obs", "sum"),
        fit_time_s=("fit_time_s", "sum"),
    ).sort_values("smape")
    g["smape_rank"] = g["smape"].rank().astype(int)
    return g.round(3)
