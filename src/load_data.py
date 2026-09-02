"""Data loading and series-selection utilities for the demand-forecasting benchmark."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "demand_forecasting_2013.csv"
STORE1_PATH = Path(__file__).resolve().parent.parent / "data" / "demand_forecasting_store1.csv"
FULL_PATH = Path(__file__).resolve().parent.parent / "data" / "demand_forecasting_full.csv"

COLUMNS = ["date", "store", "item", "sales"]


def load_data(path: str | Path = DATA_PATH, has_header: bool = True) -> pd.DataFrame:
    """Load a panel dataset with columns date, store, item, sales.

    ``has_header=False`` handles headerless exports (e.g. demand_forecasting_store1.csv).
    """
    if has_header:
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = pd.read_csv(path, header=None, names=COLUMNS, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df


def series_key(store: int | str, item: int | str) -> str:
    return f"s{int(store)}_i{int(item)}"


def add_series_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["series"] = df.apply(lambda r: series_key(r["store"], r["item"]), axis=1)
    return df


def select_sample_series(df: pd.DataFrame, per_store: int = 6, seed: int = 42) -> list[tuple[int, int]]:
    """Stratified representative sample.

    For each store, pick `per_store` items spread across the quantiles of the
    item's mean sales within that store, so the sample spans high- and low-volume
    items uniformly. Returns a list of (store, item) tuples.
    """
    chosen: list[tuple[int, int]] = []
    for store in sorted(df["store"].unique()):
        sub = df[df["store"] == store]
        means = sub.groupby("item")["sales"].mean().sort_values()
        idx = np.linspace(0, len(means) - 1, per_store).round().astype(int)
        for i in idx:
            chosen.append((int(store), int(means.index[i])))
    return chosen


def panel_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (date, series) -> sales wide frame."""
    d = add_series_key(df)
    wide = d.pivot_table(index="date", columns="series", values="sales")
    return wide
