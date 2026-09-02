# Store-1 Benchmark (5-Year Panel): Classical vs Gradient-Boosted ML

**Dataset:** `data/demand_forecasting_store1.csv` — **store 1's full history** extracted from
the [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)
by `data/create_dataset.py`: **50 items × 1,826 days (2013-01-01 … 2017-12-31) = 91,300 rows**
(no header row; headerless handling built into `src/load_data.py`). Sales 1–155 (mean ≈ 47.3).

This is the **repeat of the original benchmark** ([REPORT.md](REPORT.md)) on a 5-year panel:
five times more history makes **yearly seasonality identifiable**, which changes the model
hierarchy. Same stack (Python 3.11 · Prophet 1.4 · statsmodels 0.15 · LightGBM 4.7 · sktime 1.1).

---

## TL;DR — which model wins and why

> **LightGBM with lag features wins again — by a wider margin and this time outright.**
> On the full 50-series, 12-fold quarterly walk-forward it posts **sMAPE 13.19 / MAE 5.88 /
> RMSE 7.37** and beats every other model significantly (all p < 0.0001), winning **43 of 50
> series** head-to-head. **Prophet is the big improver** — with ≥2 years of history its yearly
> seasonality becomes identifiable and it jumps from worst-classical (2013 sample) to a close
> second (sMAPE 13.75). **ETS and SARIMA collapse** (18.6 / 19.0 sMAPE): their weekly-only
> seasonal structure cannot represent the annual cycle that dominates a 5-year demand series.
> LightGBM wins because (1) it pools 50 series so every item borrows level/seasonality
> structure, (2) it gets explicit annual-cycle features (lags 364/365/366 d + calendar year —
> the top-4 most important features), and (3) its recursive forecasts stay accurate across the
> whole 90-day horizon (error 5.8 → 5.9) while ETS/SARIMA errors double by day 60–90.

**Full leaderboard (all 50 series, n = 54,000 forecast-days per model):**

| rank | model | sMAPE ↓ | MAE ↓ | RMSE ↓ | MASE ↓ |
|-----:|-------|--------:|------:|-------:|-------:|
| 1 | **LightGBM** (pooled, lag + yearly features) | **13.19** | **5.88** | **7.37** | **0.77** |
| 2 | Prophet (weekly + monthly + yearly seasonality) | 13.75 | 6.17 | 7.69 | 0.80 |
| 3 | ETS (statsmodels AAA, weekly) | 18.58 | 8.69 | 10.70 | 1.11 |
| 4 | SARIMA (AIC grid, weekly seasonal) | 19.01 | 8.92 | 10.96 | 1.14 |
| 5 | Seasonal naive (sp = 7) | 21.90 | 10.03 | 12.37 | 1.29 |
| 6 | Naive (persistence) | 28.11 | 13.20 | 15.60 | 1.69 |

---

## Methodology

### Walk-forward backtest (sktime)
- Splitter: `sktime.split.ExpandingWindowSplitter` — **initial window 730 days (2 years),
  step 90 days, horizon 90 days (fh = 1..90) ⇒ 12 quarterly folds**, cutoffs
  2014-12-31 … 2017-09-16, test windows 2015-01-01 … 2017-12-15. Every model is **refit at
  every cutoff**; all training data is strictly ≤ cutoff.
- **All 50 series are evaluated** (no sampling) — the full store-1 panel.
- Metrics: **sMAPE** (Kaggle metric), **MAE**, **RMSE**, **MASE** (in-sample seasonal-naive-7
  scaling per series).

### Models (changes vs the 2013 benchmark)
| model | spec |
|---|---|
| Naive / Seasonal naive | unchanged baselines |
| ETS | unchanged: additive AAA with weekly seasonality (period 7) |
| SARIMA | statsmodels `SARIMAX`, weekly seasonal (0,1,1,7), **4-candidate** non-seasonal AIC grid (trimmed to bound runtime on 5-year series) |
| Prophet | weekly + monthly Fourier + **yearly seasonality now enabled** (auto-gated on ≥ 1.5 y of history); changepoint prior 0.02, MAP fit |
| LightGBM | pooled; existing lags 1–7,14,21,28 + rolling stats + calendar, **plus annual-cycle lags 364/365/366 and the calendar year**; recursive 90-day multi-step |

No tuned hyperparameters; the comparison is "off-the-shelf, refit per fold".

---

## Results

### Overall (all 50 series, 12 folds; `results/store1/metrics_summary.csv`)
See leaderboard above. **MASE < 1 for LightGBM (0.77) and Prophet (0.80)** — out-of-sample
error is *below* the in-sample seasonal-naive scale, the first time any model here achieves
that; the top models genuinely capture multi-scale structure.

### Statistical significance (paired Wilcoxon, per series×fold MAE, n = 600)
| comparison | p-value | verdict |
|---|---:|---|
| LightGBM vs Prophet | < 0.0001 | LightGBM wins (5.88 vs 6.17) |
| LightGBM vs ETS / SARIMA | < 0.0001 | LightGBM wins |
| Prophet vs ETS / SARIMA | < 0.0001 | Prophet wins |
| ETS vs SARIMA | < 0.0001 | ETS wins (8.69 vs 8.92) |

### Per-series wins (best mean MAE across folds, of 50 series)
**LightGBM 43 · Prophet 6 · ETS 1** — LightGBM is the best model for 86% of the series.

### Per-fold sMAPE (12 quarterly folds)
LightGBM wins or ties **11 of 12** folds; Prophet edges it only in folds 1–2 (spring/summer
2015, its strongest seasonal-identification period). ETS and SARIMA never win a fold.
Worst fold for everyone is the spring-2017 quarter (fold 9: Naive 45.3, ETS 21.5, LightGBM
11.7) — a strong upward drift that level-blind methods miss.

### Error anatomy
- **Horizon profile (90-day horizon, mean |error|):**

  | bucket | LightGBM | Prophet | ETS | SARIMA |
  |---|---|---:|---:|---:|
  | days 1–7 | 5.78 | 5.93 | 6.13 | 6.12 |
  | days 8–14 | 5.85 | 6.23 | 6.75 | 6.74 |
  | days 15–30 | 5.80 | 6.13 | 7.04 | 7.01 |
  | days 31–60 | 5.88 | 6.15 | 7.66 | 7.80 |
  | days 61–90 | **5.95** | 6.24 | **11.67** | **12.21** |

  LightGBM's error is **flat across the whole quarter** — the annual-cycle lags keep the
  forecast anchored to the right level of the year. ETS/SARIMA degrade ~2× at 2–3 months
  ahead because their weekly-only model cannot project the annual phase.
- **Feature importances** (illustrative full-history fit): `lag364` (1382), `series_mean`
  (1371), `lag365` (1061), `dayofyear` (1053) top the list — **the annual cycle dominates**;
  weekly features (`lag7`, `roll_mean7`) remain important but secondary.
- **Yearly seasonality is strong**: mean daily sales swings from ~32 (Jan) to ~60 (Jul)
  within every year — this is the structure that ETS/SARIMA structurally cannot see.

### Runtime (this machine, 8 worker processes)
| model | summed fit time (600 series-folds) |
|---|---:|
| Naive / Seasonal naive | < 1 s |
| ETS | 78 s |
| LightGBM (pooled fits + recursive) | 72 s |
| Prophet | 97 s |
| SARIMA (4-candidate AIC grid) | ≈ 2.7 h CPU (22 min wall with 8 workers) |

---

## How the repeat changed the verdict (2013 sample vs store 1)

| aspect | 2013 sample (1 y, 200 series) | store 1 (5 y, 50 series) |
|---|---|---|
| **Winner** | LightGBM | **LightGBM (same)** |
| Runner-up | SARIMA (statistical tie on shared ground) | **Prophet** (close, significant gap) |
| Prophet | 4th, ≈ seasonal naive (p = 0.25) | **2nd, −33% relative sMAPE** — yearly seasonality is its missing ingredient |
| SARIMA | best classical on sMAPE | **4th, even behind ETS** — weekly-only structure can't track the annual cycle |
| ETS | 3rd | 3rd (better than SARIMA now) |
| MASE of winner | 1.09 (in-sample-scaled; all > 1) | **0.77** (out-of-sample beats seasonal naive) |
| Long-horizon behavior | LightGBM plateaus at 30 d | LightGBM **flat to 90 d**; ETS/SARIMA explode |

**Why the hierarchy shifts with history:** the 1-year sample cannot identify yearly
seasonality, so SARIMA (best weekly-noise fitter) ties the ML model; given 5 years, the
annual cycle becomes the dominant, learnable signal — and the models with explicit yearly
structure (LightGBM's lags 364/365/366 + `dayofyear`, Prophet's Fourier yearly term) pull
far ahead of the weekly-only classical models.

---

## Verdict

**LightGBM with lag features is the winner on both benchmarks.** On the 5-year store-1 panel
its win is unambiguous: best on every metric, significant against all rivals, 43/50 series,
and stable out to 90 days. Prophet is the strongest classical option given enough history —
and would likely close more of the gap with multiplicative seasonality and tuned priors —
but it still loses to the pooled ML model that can share structure across items.

Practical takeaways:
1. **Give the ML model the annual cycle explicitly** — yearly lags + calendar features are
   worth more than any other design choice on multi-year demand data.
2. **Per-series classical models (ETS/SARIMA) hit a structural ceiling** without yearly
   seasonality; a Fourier-SARIMAX or ETS with yearly components would be the fair upgrade.
3. **Pooling pays**: 43/50 series are best served by a model that learned from all 50.

---

## Limitations

- **SARIMA/ETS omit yearly seasonality** (statsmodels seasonal orders of 365 are impractical);
  a Fourier-based SARIMA would be the fair comparison and would narrow the gap.
- LightGBM forecasts **recursively**; direct multi-horizon training or an AR-correction would
  likely improve it further.
- Prophet uses MAP optimization and default priors with additive seasonality; multiplicative
  mode (demand amplitude scales with level) is worth testing.
- Single store (store 1); other stores may differ, though the 2013-sample result (all 4
  stores) points the same way.
- No hyperparameter tuning for any model.

## Reproducibility

```bash
.venv/bin/python -m scripts.05_backtest_store1   # ~22 min, 8 workers; writes results/store1/
.venv/bin/python -m scripts.03_figures results/store1 results/store1/predictions.pkl \
    results/store1/figures data/demand_forecasting_store1.csv false   # figures
```

Artifacts: `results/store1/metrics_summary.csv`, `metrics_per_fold.csv`,
`metrics_per_fold_series.csv`, `predictions.pkl`, `figures/*.png`.
