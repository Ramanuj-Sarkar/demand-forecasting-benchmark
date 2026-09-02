"""Extension: LightGBM (and ML baselines) on the FULL 200-series panel.

The per-series classical models are evaluated on the 24-series stratified sample
in 02_backtest.py; pooled models can scale to the whole panel cheaply, so we run
LightGBM walk-forward on all 200 series here to show the scale-up result.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest, summarize
from src.load_data import load_data, select_sample_series

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    df = load_data()
    all_series = [(int(s), int(i)) for s, i in df[["store", "item"]].drop_duplicates().itertuples(index=False)]
    print(f"full panel: {len(all_series)} series")

    t0 = time.time()
    metrics, predictions = run_backtest(df, all_series, models=["LightGBM"], extra_pooled=True, verbose=True)
    print(f"\nfull-panel backtest finished in {time.time() - t0:.0f}s")

    summary = summarize(metrics)
    print("\n=== full-panel pooled metrics ===")
    print(summary.to_string())

    metrics.to_csv(RESULTS / "metrics_full_panel.csv", index=False)
    summary.to_csv(RESULTS / "metrics_summary_full_panel.csv")
    with open(RESULTS / "predictions_full_panel.pkl", "wb") as f:
        pickle.dump(predictions, f)
    print("\nwrote: metrics_full_panel.csv, metrics_summary_full_panel.csv, predictions_full_panel.pkl")


if __name__ == "__main__":
    main()
