"""Model implementations for the demand-forecasting benchmark.

Every per-series model exposes ``fit_predict(history, horizon) -> pd.Series``
where ``history`` is a daily ``pd.Series`` indexed by ``DatetimeIndex`` and the
returned series is indexed by the next ``horizon`` calendar days.

Pooled (ML) models expose ``fit_pooled(train_df, **kw)`` followed by
``predict_series(series_rows, horizon, **kw)`` so the same fit serves many series.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# classical, per-series models
# --------------------------------------------------------------------------- #
def ets_forecast(history: pd.Series, horizon: int, seasonal_periods: int = 7) -> pd.Series:
    """ETS (error/trend/seasonal, additive, weekly seasonality) via statsmodels.

    Falls back to Holt-Winters and then to a rolling-mean baseline on failure.
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    future = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc: pd.Series | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ETSModel(
                history, error="add", trend="add", seasonal="add",
                seasonal_periods=seasonal_periods, initialization_method="estimated",
            )
            res = model.fit(disp=False, maxiter=800)
        fc = res.forecast(horizon)
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ExponentialSmoothing(
                    history, trend="add", seasonal="add", seasonal_periods=seasonal_periods,
                ).fit()
            fc = res.forecast(horizon)
        except Exception:
            pass
    if fc is None or not np.all(np.isfinite(fc.values)):
        fc = pd.Series(np.full(horizon, float(history.mean())), index=future)
    fc = pd.Series(np.clip(fc.values, 0, None), index=future, name="sales")
    return fc


def sarima_forecast(
    history: pd.Series,
    horizon: int,
    seasonal_order: tuple = (0, 1, 1, 7),
    grid: list[tuple] | None = None,
) -> pd.Series:
    """Seasonal ARIMA with a small AIC-based order search (statsmodels SARIMAX).

    The seasonal part is fixed to (0,1,1,7) (the classic 'airline' weekly model);
    the non-seasonal part is chosen on AIC from a small grid.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    future = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    if grid is None:
        grid = SARIMA_GRID

    best_aic, best_res = np.inf, None
    for order in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    history, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                    trend=None if order[1] == 0 else None,
                )
                res = model.fit(disp=False, maxiter=100, method="lbfgs")
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic, best_res = res.aic, res
        except Exception:
            continue

    if best_res is None:
        return pd.Series(_seasonal_naive(history.values, horizon, 7), index=future, name="sales")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = best_res.forecast(horizon)
    fc = pd.Series(np.clip(fc.values, 0, None), index=future, name="sales")
    return fc


def prophet_forecast(
    history: pd.Series,
    horizon: int,
    changepoint_prior_scale: float = 0.02,
    seasonality_mode: str = "additive",
) -> pd.Series:
    """Prophet with weekly + monthly Fourier seasonality.

    Yearly seasonality is enabled only when the history is long enough to
    identify it (>= 1.5 years); with ~1 year of data it is turned off because
    it cannot be identified and only overfits.
    """
    from prophet import Prophet

    future = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    df = pd.DataFrame({"ds": history.index, "y": history.values})
    n_years = (history.index[-1] - history.index[0]).days / 365.25
    m = Prophet(
        growth="linear",
        weekly_seasonality=True,
        yearly_seasonality=n_years >= 1.5,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        uncertainty_samples=0,
    )
    m.add_seasonality(name="monthly", period=30.44, fourier_order=3)
    try:
        m.fit(df)
        future_df = m.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        yhat = m.predict(future_df)["yhat"].values
        if not np.all(np.isfinite(yhat)):
            raise RuntimeError("non-finite prophet forecast")
    except Exception:
        return seasonal_naive_forecast(history, horizon)
    return pd.Series(np.clip(yhat, 0, None), index=future, name="sales")


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def naive_forecast(history: pd.Series, horizon: int) -> pd.Series:
    future = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series(np.full(horizon, float(history.iloc[-1])), index=future, name="sales")


def _seasonal_naive(vals: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    """Same-weekday-last-week persistence forecast, wrapping for long horizons."""
    n = len(vals)
    fc = np.empty(horizon)
    for i in range(horizon):
        idx = n - period + (i % period)
        fc[i] = vals[idx] if idx >= 0 else float(np.mean(vals))
    return fc


def seasonal_naive_forecast(history: pd.Series, horizon: int, period: int = 7) -> pd.Series:
    future = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc = _seasonal_naive(history.values, horizon, period)
    return pd.Series(np.clip(fc, 0, None), index=future, name="sales")


# --------------------------------------------------------------------------- #
# pooled ML models with lag features
# --------------------------------------------------------------------------- #
LAG_DAYS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
YEARLY_LAG_DAYS = [364, 365, 366]  # annual-cycle lags, usable once >= 2 years of history
ROLL_WINDOWS = [7, 14, 28]
SARIMA_GRID = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (0, 1, 2)]


def feature_columns(yearly_lags: bool) -> list[str]:
    cols = (
        [f"lag{l}" for l in LAG_DAYS]
        + [f"roll_mean{w}" for w in ROLL_WINDOWS]
        + [f"roll_std{w}" for w in ROLL_WINDOWS]
        + ["series_mean", "dow", "dom", "month", "weekofyear", "dayofyear", "is_weekend", "store", "item"]
    )
    if yearly_lags:
        cols = [f"lag{l}" for l in YEARLY_LAG_DAYS] + cols + ["year"]
    return cols


def build_lag_features(df: pd.DataFrame, yearly_lags: bool = False) -> pd.DataFrame:
    """Add lag/rolling/calendar/static features to the panel (no leakage).

    All lag and rolling features are shifted by one day, so a row's features use
    only information available strictly before that date. With ``yearly_lags``,
    annual-cycle lags (364/365/366 days) and the calendar year are added; rows
    without a full prior year (NaN yearly lags) are dropped downstream.
    """
    d = df.sort_values(["store", "item", "date"]).copy()
    g = d.groupby(["store", "item"], group_keys=False)

    for lag in LAG_DAYS:
        d[f"lag{lag}"] = g["sales"].shift(lag)
    if yearly_lags:
        for lag in YEARLY_LAG_DAYS:
            d[f"lag{lag}"] = g["sales"].shift(lag)
        d["year"] = d["date"].dt.year - d["date"].dt.year.min()
    for w in ROLL_WINDOWS:
        shifted = g["sales"].shift(1)
        d[f"roll_mean{w}"] = shifted.rolling(w).mean()
        d[f"roll_std{w}"] = shifted.rolling(w).std()
    # expanding mean of the series itself (level anchor, strict past only)
    d["series_mean"] = g["sales"].transform(lambda s: s.shift(1).expanding().mean())

    # calendar features
    d["dow"] = d["date"].dt.dayofweek
    d["dom"] = d["date"].dt.day
    d["month"] = d["date"].dt.month
    d["weekofyear"] = d["date"].dt.isocalendar().week.astype(int)
    d["dayofyear"] = d["date"].dt.dayofyear
    d["is_weekend"] = (d["dow"] >= 5).astype(int)

    # static features
    d["store"] = d["store"].astype(int)
    d["item"] = d["item"].astype(int)
    return d


def _temporal_split(train: pd.DataFrame, val_days: int = 30, seed: int = 42):
    """Temporal holdout inside the training window: last val_days per series."""
    cutoff = train["date"].max() - pd.Timedelta(days=val_days)
    tr = train[train["date"] <= cutoff]
    va = train[train["date"] > cutoff]
    return tr, va


def fit_lightgbm(train: pd.DataFrame, params: dict | None = None, early_stopping: bool = True,
                 yearly_lags: bool = False):
    """Fit a pooled LightGBM regressor on lag features with temporal early stopping."""
    import lightgbm as lgb

    cols = feature_columns(yearly_lags)
    feats = build_lag_features(train, yearly_lags=yearly_lags)
    feats = feats.dropna(subset=cols).reset_index(drop=True)
    X = feats[cols]
    y = feats["sales"]

    if params is None:
        params = dict(
            objective="regression",
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    if early_stopping and len(X) > 3000:
        tr, va = _temporal_split(feats)
        model = lgb.LGBMRegressor(**params)
        model.fit(
            tr[cols], tr["sales"],
            eval_set=[(va[cols], va["sales"])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        params = {**params, "n_estimators": min(params.get("n_estimators", 800), 800)}
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
    return model


def _row_features(date: pd.Timestamp, extended: np.ndarray, series_mean: float,
                  store: int, item: int, yearly_lags: bool = False,
                  base_year: int | None = None) -> dict:
    """Feature row for one forecast date given all observed+predicted values.

    ``extended`` holds the observed history followed by predictions already made;
    its last element is the value one day before ``date``.
    """
    f: dict = {}
    n = len(extended)
    for lag in LAG_DAYS:
        f[f"lag{lag}"] = extended[n - lag] if n - lag >= 0 else np.nan
    if yearly_lags:
        for lag in YEARLY_LAG_DAYS:
            f[f"lag{lag}"] = extended[n - lag] if n - lag >= 0 else np.nan
        f["year"] = date.year - (base_year if base_year is not None else date.year)
    for w in ROLL_WINDOWS:
        win = extended[n - w:] if n >= w else extended
        f[f"roll_mean{w}"] = float(np.mean(win))
        f[f"roll_std{w}"] = float(np.std(win)) if len(win) > 1 else 0.0
    f["series_mean"] = series_mean
    f["dow"] = date.dayofweek
    f["dom"] = date.day
    f["month"] = date.month
    f["weekofyear"] = date.isocalendar().week
    f["dayofyear"] = date.dayofyear
    f["is_weekend"] = int(date.dayofweek >= 5)
    f["store"] = store
    f["item"] = item
    return f


def predict_with_lags(model, series_df: pd.DataFrame, cutoff: pd.Timestamp, horizon: int,
                      yearly_lags: bool = False) -> pd.Series:
    """Recursive multi-step forecast for one series using its own history."""
    hist = series_df[series_df["date"] <= cutoff].sort_values("date")
    vals = hist["sales"].values
    series_mean = float(vals.mean())
    store, item = int(hist["store"].iloc[0]), int(hist["item"].iloc[0])
    base_year = int(hist["date"].dt.year.min())
    extended = list(vals)
    future = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    cols = feature_columns(yearly_lags)
    preds = []
    for tdate in future:
        row = _row_features(tdate, np.array(extended), series_mean, store, item,
                            yearly_lags=yearly_lags, base_year=base_year)
        p = float(model.predict(pd.DataFrame([row])[cols])[0])
        p = max(p, 0.0)
        preds.append(p)
        extended.append(p)
    return pd.Series(preds, index=future, name="sales")


# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #
PER_SERIES_MODELS = {
    "Naive": naive_forecast,
    "SeasonalNaive": seasonal_naive_forecast,
    "ETS": ets_forecast,
    "SARIMA": sarima_forecast,
    "Prophet": prophet_forecast,
}

POOLED_MODELS = {
    "LightGBM": fit_lightgbm,
}
