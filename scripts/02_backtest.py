"""Run the walk-forward backtest and persist metrics + predictions."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest, summarize
from src.load_data import load_data, select_sample_series

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

PER_STORE = 6  # -> 24-series stratified sample


def main() -> None:
    df = load_data()
    sample = select_sample_series(df, per_store=PER_STORE, seed=42)
    print(f"sample series ({len(sample)}): {sample}")

    t0 = time.time()
    metrics, predictions = run_backtest(df, sample, verbose=True)
    print(f"\nbacktest finished in {time.time() - t0:.0f}s")

    summary = summarize(metrics)
    print("\n=== pooled metrics (per model) ===")
    print(summary.to_string())

    metrics.to_csv(RESULTS / "metrics_per_fold_series.csv", index=False)
    summary.to_csv(RESULTS / "metrics_summary.csv")
    with open(RESULTS / "predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)
    # also save per-fold aggregate for easy reading
    per_fold = metrics.groupby(["model", "fold"]).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"), mase=("mase", "mean"),
    ).reset_index()
    per_fold.to_csv(RESULTS / "metrics_per_fold.csv", index=False)
    print("\nwrote: metrics_summary.csv, metrics_per_fold.csv, metrics_per_fold_series.csv, predictions.pkl")


if __name__ == "__main__":
    main()
