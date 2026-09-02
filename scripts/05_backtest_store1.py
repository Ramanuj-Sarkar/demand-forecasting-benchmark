"""Store-1 benchmark: repeat the walk-forward comparison on the 5-year store-1 panel.

Data: data/demand_forecasting_store1.csv (store 1, 50 items, 2013-01-01..2017-12-31).
Protocol (scaled to multi-year data): sktime ExpandingWindowSplitter with an
initial window of 2 years, 90-day step and 90-day horizon -> 12 quarterly folds;
every model refit at every cutoff. Prophet gains yearly seasonality and LightGBM
gains annual-cycle lag features (364/365/366d + year). All 50 series are evaluated
(the full panel, not a sample). Results go to results/store1/.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import pandas as pd

from src import models
from src.backtest import run_backtest, summarize
from src.load_data import STORE1_PATH, load_data

RESULTS = Path(__file__).resolve().parent.parent / "results" / "store1"
RESULTS.mkdir(parents=True, exist_ok=True)

INITIAL_WINDOW = 730   # 2 years
STEP_LENGTH = 90       # quarterly evaluation
HORIZON = 90           # 3-month-ahead horizon (the competition's framing)
N_JOBS = min(8, os.cpu_count() or 1)


def main() -> None:
    df = load_data(STORE1_PATH, has_header=False)
    print(f"store1 panel: rows={len(df):,}  series={df['store'].nunique() * df['item'].nunique()}  "
          f"dates={df['date'].nunique()}  ({df['date'].min().date()}..{df['date'].max().date()})")

    all_series = [(int(s), int(i)) for s, i in df[["store", "item"]].drop_duplicates().itertuples(index=False)]
    print(f"evaluating all {len(all_series)} series")

    # trim the SARIMA AIC grid to bound runtime on the longer series
    models.SARIMA_GRID = [(0, 1, 1), (1, 0, 1), (1, 1, 1), (2, 1, 1)]

    t0 = time.time()
    metrics, predictions = run_backtest(
        df, all_series, verbose=True,
        initial_window=INITIAL_WINDOW, step_length=STEP_LENGTH, horizon=HORIZON,
        n_jobs=N_JOBS,
    )
    print(f"\nstore1 backtest finished in {time.time() - t0:.0f}s (n_jobs={N_JOBS})")

    summary = summarize(metrics)
    print("\n=== pooled metrics (per model) ===")
    print(summary.to_string())

    metrics.to_csv(RESULTS / "metrics_per_fold_series.csv", index=False)
    summary.to_csv(RESULTS / "metrics_summary.csv")
    per_fold = metrics.groupby(["model", "fold"]).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"), mase=("mase", "mean"),
    ).reset_index()
    per_fold.to_csv(RESULTS / "metrics_per_fold.csv", index=False)
    with open(RESULTS / "predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)
    print(f"\nwrote to {RESULTS}: metrics_summary.csv, metrics_per_fold.csv, "
          f"metrics_per_fold_series.csv, predictions.pkl")


if __name__ == "__main__":
    main()
