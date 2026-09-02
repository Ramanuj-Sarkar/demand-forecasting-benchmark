# Demand Forecasting Benchmark: Classical vs Gradient-Boosted ML

**Dataset:** `data/demand_forecasting_2013.csv` — a deliberately chosen one-year sample (2013) of the
[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only):
**4 stores × 50 items = 200 daily sales series × 365 days = 73,000 rows.**
Sales range 1–169 (mean ≈ 47.6, std ≈ 24.5).

> **Repeat on 5 years of data:** the same benchmark was re-run on the full store-1 panel
> (2013–2017, `data/demand_forecasting_store1.csv`) in **[REPORT_STORE1.md](REPORT_STORE1.md)** —
> LightGBM wins there too, and Prophet jumps to second once yearly seasonality is identifiable.

**Stack:** Python 3.11 · Prophet 1.4 · statsmodels 0.15 (ETS + SARIMA) · LightGBM 4.7 ·
sktime 1.1 (walk-forward splitter + metrics).

---

## TL;DR — which model wins and why

> **LightGBM with lag features wins.** On the same 24-series evaluation ground it is
> **statistically tied with SARIMA** when trained only on those 24 series (MAE 8.29 vs 8.50,
> p ≈ 0.48), and **pulls clearly ahead once it can pool the full 200-series panel**:
> MAE 8.01 vs 8.50 (p = 0.0012), sMAPE 18.68 vs 18.84, RMSE 9.55 vs 10.04.
> It also wins more per-series head-to-heads (9 of 24), degrades less at horizons > 2 weeks
> (its error plateaus while SARIMA's keeps growing), and absorbs the hard December fold
> better than the weekly-seasonal classical models. **Why:** pooled cross-series learning +
> calendar features capture the level and the strong weekly/end-of-year demand structure
> that per-series ETS/ARIMA (weekly seasonality only) cannot see in a single noisy 365-day
> series. Prophet under-delivers here because one year of history cannot identify yearly
> seasonality — its advantage over a seasonal naive baseline is not significant (p = 0.25).

**Full leaderboard (24-series shared ground, n = 4,320 forecasts per model):**

| rank | model | sMAPE ↓ | MAE ↓ | RMSE ↓ | MASE ↓ |
|-----:|-------|--------:|------:|-------:|-------:|
| 1 | **LightGBM** (trained on all 200 series) | **18.68** | **8.01** | **9.55** | — |
| 2 | SARIMA (statsmodels, AIC order search) | 18.84 | 8.50 | 10.04 | 1.09 |
| 3 | LightGBM (trained on the 24 series only) | 19.08 | 8.29 | 9.87 | 1.10 |
| 4 | ETS (statsmodels, additive AAA, weekly) | 19.89 | 9.02 | 10.55 | 1.16 |
| 5 | Prophet (weekly + monthly Fourier) | 20.59 | 9.48 | 11.01 | 1.22 |
| 6 | Seasonal naive (same weekday last week) | 21.45 | 9.64 | 11.70 | 1.25 |
| 7 | Naive (persistence) | 23.59 | 10.98 | 12.98 | 1.39 |

On the **full 200-series panel**, LightGBM reaches sMAPE **17.91** / MAE **8.02** / RMSE **9.59**.

---

## Methodology

### Walk-forward backtest (sktime)
- Splitter: `sktime.split.ExpandingWindowSplitter` — **initial window 180 days, step 30 days,
  forecast horizon 30 days (fh = 1..30) ⇒ 6 folds** (cutoffs 2013-06-29 … 2013-11-26;
  test windows 2013-06-30 … 2013-12-26).
- **Every model is refit at every fold cutoff** — no look-ahead; all training data is
  strictly ≤ cutoff.
- Per-series (ETS, SARIMA, Prophet) and pooled (LightGBM) models are evaluated on the **same
  series and the same folds**; per-series forecasts are 30-day-ahead, per-fold.
- Metrics: **sMAPE** (the Kaggle competition metric), **MAE**, **RMSE**, and **MASE**
  (scaled by each series' in-sample seasonal-naive-7 MAE).

### Evaluation sample
Classical per-series models are fitted 200×6 = 1,200 times per model; to keep the run
practical, the head-to-head is run on a **stratified sample of 24 series** (6 items per
store, spread across the quantiles of mean sales so low- and high-volume items are both
represented). The pooled LightGBM is additionally run on **all 200 series** (a) to show the
scale-up and (b) to be re-evaluated *on the same 24 series* for a strict apples-to-apples
comparison.

### Models
| model | family | spec |
|---|---|---|
| Naive | baseline | last observed value |
| Seasonal naive (sp=7) | baseline | same weekday last week |
| ETS | classical | statsmodels `ETSModel`, additive error/trend/seasonal, weekly seasonality (period 7), estimated initialization |
| SARIMA | classical | statsmodels `SARIMAX`, seasonal order fixed to (0,1,1,7) ("airline" weekly model), non-seasonal order chosen per series/fold on AIC from a 6-candidate grid |
| Prophet | classical (Bayesian) | weekly seasonality + custom monthly Fourier (period 30.44, order 3); yearly disabled (not identifiable in 1 year); changepoint prior 0.02, MAP fit |
| LightGBM | gradient boosting (pooled) | regression, ~800–2000 trees, lr 0.05, temporal-holdout early stopping; features: lags 1–7,14,21,28 · rolling mean/std 7/14/28 · expanding series mean · calendar (dow, dom, month, week, dayofyear, weekend) · store/item ids; recursive 30-day multi-step |

No model has tuned hyperparameters beyond these sensible defaults (SARIMA's order is the
only auto-selected component) — the comparison is "off-the-shelf, refit per fold".

---

## Results

### Overall (shared 24-series ground)
Pooled across all 6 folds × 24 series (4,320 forecast-days per model). See
`results/metrics_summary.csv` and `results/metrics_per_fold_series.csv`.

| model | sMAPE | MAE | RMSE | MASE |
|---|---:|---:|---:|---:|
| **LightGBM-200** (pooled over 200 series) | **18.68** | **8.01** | **9.55** | — |
| SARIMA | 18.84 | 8.50 | 10.04 | 1.09 |
| LightGBM-24 (pooled over the sample) | 19.08 | 8.29 | 9.87 | 1.10 |
| ETS | 19.89 | 9.02 | 10.55 | 1.16 |
| Prophet | 20.59 | 9.48 | 11.01 | 1.22 |
| Seasonal naive | 21.45 | 9.64 | 11.70 | 1.25 |
| Naive | 23.59 | 10.98 | 12.98 | 1.39 |

### Statistical significance (paired Wilcoxon signed-rank, per series×fold MAE, n = 144)
| comparison | p-value | verdict |
|---|---:|---|
| LightGBM-24 vs SARIMA (MAE) | 0.48 | **statistical tie** |
| LightGBM-24 vs SARIMA (sMAPE) | 0.48 | **statistical tie** |
| **LightGBM-200 vs SARIMA (MAE)** | **0.0012** | **LightGBM wins** |
| LightGBM-24 vs ETS / Prophet | < 0.01 | LightGBM wins |
| SARIMA vs ETS / Prophet | < 0.001 | SARIMA wins |
| ETS vs Prophet | 0.023 | ETS wins |
| Prophet vs seasonal naive | 0.25 | no significant advantage |

### Per-series wins (best mean MAE per series, 24-series run)
**LightGBM 9 · SARIMA 8 · Prophet 5 · ETS 2** — the top two split the series between them;
no classical model dominates any series class.

### Error anatomy
- **Horizon profile** (mean |error| by steps-ahead): days 1–7 → SARIMA 7.17 vs LightGBM
  7.32 (near-tie); days 8–14 → tie (8.60 vs 8.59); days 15–21 → LightGBM 8.48 vs SARIMA
  8.77; days 22–30 → LightGBM 8.69 vs SARIMA 9.25. **SARIMA is marginally better in the
  first week; LightGBM is better from week 2 on** — its recursive error growth flattens,
  SARIMA's keeps compounding.
- **The December fold (fold 5) is hard for everyone**: sMAPE 28–35 vs ~12–22 elsewhere,
  because December demand drops sharply (mean daily sales 35.8 vs July's 60.4 peak) after
  months of mid-year strength. The models that anticipate the drop best are the
  **calendar-aware** ones — Prophet (monthly Fourier) 28.4 and LightGBM (`dayofyear`/`dom`
  features) 28.9 — while weekly-seasonal SARIMA (32.8) and ETS (34.5) over-forecast December.
- **Weekday structure**: demand rises monotonically Mon→Sun (37.8 → 56.5); lag-7 and rolling
  7-day features are the most-used LightGBM inputs after the series level (`series_mean`,
  `lag7`, `roll_mean7`, `lag1`, `lag14` in the top 5).

### Runtime (this machine, single process)
| model | total fit+predict time (24-series run) |
|---|---:|
| Naive / Seasonal naive | < 0.1 s |
| ETS | 5.4 s |
| LightGBM (pooled, 24 series) | 3.1 s |
| Prophet | 5.6 s |
| SARIMA (AIC grid) | 82.7 s |
| Full 200-series LightGBM run | 33 s |

---

## Verdict and why

**Winner: LightGBM with lag features, trained on the pooled panel.**

1. **It ties the best classical model with equal data.** Trained only on the same 24 series
   it is statistically indistinguishable from SARIMA (p ≈ 0.48) — meaning the per-series
   SARIMA is a very strong opponent on short noisy daily series.
2. **It wins decisively with full data.** Pooling 200 series lets LightGBM borrow level and
   seasonality structure across items/stores; on the identical evaluation ground it beats
   SARIMA on MAE with p = 0.0012 (~6% lower error), and on sMAPE and RMSE too. No classical
   model can use other series' data — each of its 200 fits sees one noisy 365-day series.
3. **Its error profile is better where it matters operationally.** Beyond 2 weeks ahead its
   recursive error plateaus while SARIMA's compounds, and it absorbs regime changes (the
   December drop) via calendar features.
4. **Prophet under-delivers in this setting.** One year of daily history cannot identify
   yearly seasonality (turned off here); its residual weekly/monthly Fourier model is not
   significantly better than a seasonal naive baseline. Prophet shines with multi-year
   histories and holiday/regressor support, neither of which applies to this sample.
5. **ETS/ARIMA remain attractive operationally**: they need no feature engineering, fit in
   seconds per series, and SARIMA in particular is a top-tier, fully automatic baseline that
   rivals LightGBM on a per-series basis. The ML win comes from scale, not per-series skill.

---

## Limitations

- **One year of data**: no yearly seasonality is identifiable, and there is no out-of-year
  test; the December fold is the only regime-change test.
- **Classical models evaluated on a 24-series sample** (cost of ~1,200 refits); the pooled
  model is evaluated on the full panel, so full-panel vs full-panel comparisons are only
  available for LightGBM. The shared-ground head-to-head is strictly apples-to-apples.
- **LightGBM forecasts recursively** (predictions feed back as lags); error accumulation is
  bounded but a direct multi-step or auto-regressive correction could improve it further.
- **Hyperparameters are defaults** except SARIMA's AIC order search; a tuned LightGBM
  (feature selection, multi-horizon targets) would likely widen the gap, and a tuned
  Prophet with more history would narrow it.
- MASE > 1 for all models: out-of-sample error exceeds the in-sample seasonal-naive scale —
  expected for noisy daily demand; use it for relative, not absolute, judgment.

---

## Reproducibility

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.01_eda          # data exploration -> results/figures/eda_*.png
.venv/bin/python -m scripts.02_backtest     # 24-series walk-forward -> results/metrics_*.csv, predictions.pkl
.venv/bin/python -m scripts.04_full_panel   # LightGBM on all 200 series
.venv/bin/python -m scripts.03_figures      # report figures -> results/figures/
```

Artifacts: `results/metrics_summary.csv`, `results/metrics_per_fold.csv`,
`results/metrics_per_fold_series.csv`, `results/metrics_full_panel.csv`,
`results/predictions.pkl`, `results/predictions_full_panel.pkl`, `results/figures/*.png`.
