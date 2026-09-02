"""Produce report figures from the backtest results.

Usage:
    python -m scripts.03_figures [results_dir] [predictions_pkl] [fig_dir] [data_path] [has_header]

Defaults target the original 2013 benchmark (results/, predictions.pkl); pass
results/store1 predictions.pkl etc. for the store-1 benchmark.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.load_data import load_data

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
PRED_PKL = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "predictions.pkl"
FIG = Path(sys.argv[3]) if len(sys.argv) > 3 else RESULTS / "figures"
DATA_PATH = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "data" / "demand_forecasting_2013.csv"
HAS_HEADER = (sys.argv[5].lower() != "false") if len(sys.argv) > 5 else True
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

ORDER = ["Naive", "SeasonalNaive", "ETS", "SARIMA", "Prophet", "LightGBM"]
COLORS = {
    "Naive": "#9e9e9e", "SeasonalNaive": "#757575", "ETS": "#1f77b4",
    "SARIMA": "#ff7f0e", "Prophet": "#2ca02c", "LightGBM": "#d62728",
}


def main() -> None:
    metrics = pd.read_csv(RESULTS / "metrics_per_fold_series.csv")
    per_fold = pd.read_csv(RESULTS / "metrics_per_fold.csv")
    with open(PRED_PKL, "rb") as f:
        predictions = pickle.load(f)
    horizon = max(len(p) for d in predictions.values() for p in d.values())
    steps_label = f"each fold = next {horizon} days"

    # --- 1. model ranking bar chart (SMAPE + MAE) ---
    order = [m for m in ORDER if m in metrics["model"].unique()]
    order += [m for m in metrics["model"].unique() if m not in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in zip(axes, ["smape", "mae"], ["sMAPE (%)", "MAE (units)"]):
        vals = metrics.groupby("model")[metric].mean().reindex(order)
        ax.bar(vals.index, vals.values, color=[COLORS.get(m, "#333") for m in vals.index])
        ax.set_title(f"Overall {title} (lower is better)")
        for i, v in enumerate(vals.values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, vals.max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIG / "model_ranking.png", bbox_inches="tight")
    plt.close(fig)

    # --- 2. per-fold SMAPE trajectories ---
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in order:
        sub = per_fold[per_fold["model"] == m].sort_values("fold")
        ax.plot(sub["fold"], sub["smape"], marker="o", ms=4, label=m, color=COLORS.get(m))
    ax.set_xlabel(f"fold ({steps_label})")
    ax.set_ylabel("sMAPE (%)")
    ax.set_title("sMAPE by backtest fold")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "smape_by_fold.png", bbox_inches="tight")
    plt.close(fig)

    # --- 3. error by forecast horizon (steps-ahead) ---
    df3 = load_data(DATA_PATH, has_header=HAS_HEADER)
    df3["series"] = df3["store"].astype(str).map(lambda s: f"s{s}") + "_i" + df3["item"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in order:
        errs = []
        for (mod, fold), preds_m in predictions.items():
            if mod != m:
                continue
            for s, p in preds_m.items():
                act = df3[df3["series"] == s].set_index("date")["sales"].reindex(p.index).values
                pv = p.values
                mask = ~np.isnan(act)
                if mask.sum() == 0:
                    continue
                errs.append(np.abs(act - pv))
        if not errs:
            continue
        Y = np.vstack(errs)
        steps = np.arange(1, Y.shape[1] + 1)
        ax.plot(steps, Y.mean(axis=0), label=m, color=COLORS.get(m))
    ax.set_xlabel("steps ahead")
    ax.set_ylabel("mean |error|")
    ax.set_title("Mean absolute error by forecast horizon")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "error_by_horizon.png", bbox_inches="tight")
    plt.close(fig)

    # --- 4. example forecasts vs actuals (last fold, representative series) ---
    df = load_data(DATA_PATH, has_header=HAS_HEADER)
    df["series"] = df["store"].astype(str).map(lambda s: f"s{s}") + "_i" + df["item"].astype(str)
    last_fold = max(f for _, f in predictions.keys())
    # pick the series with median total demand
    sample_series = sorted({s for (_, f), d in predictions.items() if f == last_fold for s in d})
    vols = df.groupby("series")["sales"].sum()
    pick = sample_series[np.argsort([vols[s] for s in sample_series])[len(sample_series) // 2]]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    actual = df[df["series"] == pick]
    test_dates = sorted(predictions[(ORDER[0], last_fold)][pick].index)
    cut = test_dates[0] - pd.Timedelta(days=28)
    hist = actual[(actual["date"] >= cut) & (actual["date"] < test_dates[0])]
    ax.plot(hist["date"], hist["sales"], color="#888", lw=0.8, label="history (last 4 weeks)")
    ax.plot(test_dates, actual[actual["date"].isin(test_dates)]["sales"].values, color="k", lw=1.4, label="actual")
    for m in ORDER:
        if (m, last_fold) not in predictions or pick not in predictions[(m, last_fold)]:
            continue
        p = predictions[(m, last_fold)][pick]
        ax.plot(p.index, p.values, lw=1.1, alpha=0.9, label=m, color=COLORS.get(m))
    ax.set_title(f"Forecasts vs actual, series {pick}, fold {last_fold} ({horizon}-day horizon)")
    ax.set_ylabel("sales")
    ax.legend(ncol=4, fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "forecast_example.png", bbox_inches="tight")
    plt.close(fig)

    # --- 5. per-series MAE boxplots ---
    fig, ax = plt.subplots(figsize=(9, 4.4))
    data = [metrics[metrics["model"] == m]["mae"].values for m in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("MAE (units)")
    ax.set_title("Distribution of per-series MAE (across folds)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "mae_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
