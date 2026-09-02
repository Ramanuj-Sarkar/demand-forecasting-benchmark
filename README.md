# Demand Forecasting Benchmark

Benchmark of **classical** (ARIMA/ETS via Prophet and statsmodels) vs **gradient-boosted**
(LightGBM with lag features) forecasting, using a **walk-forward backtest** built on sktime.

Two datasets from the [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only):

| dataset | file | span | series | rows |
|---|---|---|---|---|
| 2013 sample | `data/demand_forecasting_2013.csv` | 2013 (365 d) | 4 stores × 50 items = 200 | 73,000 |
| Store 1 | `data/demand_forecasting_store1.csv` | 2013–2017 (1826 d) | store 1 × 50 items = 50 | 91,299 |

(`demand_forecasting_store1.csv` is generated from the full Kaggle file
`demand_forecasting_full.csv` by `data/create_dataset.py` and has no header row.)

**Results**
- 1-year sample: **LightGBM wins** — tied with SARIMA on shared data, ahead when pooled over
  all 200 series (MAE 8.01 vs 8.50, p = 0.0012). See [REPORT.md](REPORT.md).
- 5-year store 1: **Prophet wins** — yearly seasonality becomes identifiable; see
  [REPORT_STORE1.md](REPORT_STORE1.md).

## Layout

```
data/                                  # demand_forecasting_2013.csv, _store1.csv, _full.csv
src/load_data.py                       # loading (header/no-header), series keys, sampling
src/models.py                          # ETS, SARIMA, Prophet, Naive, LightGBM-lag wrappers
src/backtest.py                        # sktime ExpandingWindowSplitter harness (parallel)
scripts/01_eda.py                      # exploration figures
scripts/02_backtest.py                 # 24-series head-to-head on the 2013 sample
scripts/04_full_panel.py               # LightGBM on all 200 series (2013 sample)
scripts/05_backtest_store1.py          # full 50-series backtest on 5-year store-1 panel
scripts/03_figures.py                  # report figures (arg-driven, both benchmarks)
results/                               # 2013-sample results + figures
results/store1/                        # store-1 results + figures
REPORT.md / REPORT_STORE1.md           # methodology, results, verdicts, limitations
```

## Setup (macOS, Python 3.11)

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt   # needs libomp for LightGBM (brew install libomp)
```

## Run

```bash
.venv/bin/python -m scripts.01_eda              # EDA figures
.venv/bin/python -m scripts.02_backtest         # 2013 sample, 6-fold walk-forward (~2 min)
.venv/bin/python -m scripts.04_full_panel       # LightGBM on all 200 series (~1 min)
.venv/bin/python -m scripts.05_backtest_store1  # store 1, 12 quarterly folds (~10-15 min)
.venv/bin/python -m scripts.03_figures          # 2013-sample figures
.venv/bin/python -m scripts.03_figures results/store1 results/store1/predictions.pkl \
    results/store1/figures data/demand_forecasting_store1.csv false   # store-1 figures
```

Set `MPLCONFIGDIR` to a writable dir if `~/.matplotlib` is not writable.

## Protocol

- sktime `ExpandingWindowSplitter`, every model refit at every cutoff (no look-ahead):
  - 2013 sample: initial 180 d, step 30 d, horizon 30 d → 6 folds (24-series stratified sample for the classical models).
  - store 1: initial 730 d (2 y), step 90 d, horizon 90 d → 12 quarterly folds (all 50 series).
- Metrics: sMAPE (Kaggle's metric), MAE, RMSE, MASE (seasonal-naive-7 scaled).
- Prophet enables yearly seasonality only with ≥ ~2 years of history; LightGBM gains
  annual-cycle lags (364/365/366 d) + calendar year on the multi-year panel.
