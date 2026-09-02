"""EDA for demand_forecasting_2013.csv: structure, seasonality, and series variety."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.load_data import load_data, select_sample_series

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def main() -> None:
    df = load_data()
    print(f"rows={len(df):,}  stores={df['store'].nunique()}  items={df['item'].nunique()}  "
          f"dates={df['date'].nunique()}  ({df['date'].min().date()}..{df['date'].max().date()})")
    print(df.describe().round(2).to_string())

    daily = df.groupby("date")["sales"].sum()
    by_dow = df.groupby(df["date"].dt.dayofweek)["sales"].mean()
    by_month = df.groupby(df["date"].dt.month)["sales"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(daily.index, daily.values, lw=0.8, alpha=0.9)
    axes[0, 0].set_title("Total daily demand (all 200 series)")
    axes[0, 1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], by_dow.values)
    axes[0, 1].set_title("Mean sales by weekday")
    axes[1, 0].bar(range(1, 13), by_month.values)
    axes[1, 0].set_title("Mean sales by month")
    axes[1, 1].hist(df["sales"], bins=50, log=True)
    axes[1, 1].set_title("Sales distribution (log-y)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_overview.png", bbox_inches="tight")
    plt.close(fig)

    # four example series, one per store
    sample = select_sample_series(df, per_store=1, seed=7)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (store, item) in zip(axes.ravel(), sample):
        sub = df[(df["store"] == store) & (df["item"] == item)].set_index("date")["sales"]
        ax.plot(sub.index, sub.values, lw=0.7)
        ax.set_title(f"store={store} item={item}  mean={sub.mean():.1f}")
        ax.set_ylabel("sales")
    fig.suptitle("Example series (one per store)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "eda_example_series.png", bbox_inches="tight")
    plt.close(fig)

    # store x item mean-sales heatmap
    pivot = df.pivot_table(index="store", columns="item", values="sales", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 2.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(pivot.shape[0]), [f"store {s}" for s in pivot.index])
    ax.set_xlabel("item")
    ax.set_title("Mean sales by store x item")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "eda_store_item_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
