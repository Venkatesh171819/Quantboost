# QuantBoost

A quantitative research terminal for regime-aware, gradient-boosted equity forecasting — built as a Streamlit app backed by a shared research pipeline in `src/quantboost/`.

QuantBoost turns daily price panels into vol-normalised forward-return forecasts, validates them with purged walk-forward cross-validation, and converts the scores into a cost- and risk-budgeted long/short book. The app, the notebooks, and `scripts/run_pipeline.py` all run the exact same pipeline code, so nothing in the UI is presentation-only math.

> Educational research code, not investment advice.

## Highlights

- **Cross-sectional signal.** Momentum, mean-reversion, volatility, liquidity, and fractional-differencing features, all standardised cross-sectionally per date.
- **Multiple engines.** XGBoost, CatBoost, and AdaBoost trained side by side, plus an ensemble — AdaBoost is kept in deliberately as a fat-tail-sensitive baseline.
- **Purged walk-forward validation.** Expanding-window folds with embargoed, purged label overlap; imputation, scaling, and hyperparameter tuning (Optuna, maximising rank IC) all happen inside the training folds.
- **Honest performance reporting.** Deflated Sharpe Ratio, cost sensitivity out to 5x, factor attribution against market and momentum, and a simulated-data null test (regime-switching GBM) to catch pipeline leakage.
- **Risk and execution.** Triple-barrier labeling, GARCH/EGARCH/GJR conditional volatility, Kelly-fraction sizing, VaR/ES with Kupiec and Christoffersen tests, half-spread + square-root market impact costs, and vol-targeted leverage.
- **Works offline.** Falls back to a simulated price panel if `yfinance` isn't installed or live data isn't available.

## Project structure

```
app.py                  # Streamlit UI — presentation only
src/quantboost/         # the research pipeline (data, features, labeling, models,
                         #   tuning, backtest, portfolio, risk, tsa, config)
scripts/run_pipeline.py # headless CLI entry point (same pipeline as the app)
notebooks/               # exploratory notebooks
tests/                   # pytest suite
reports/                 # generated output (csv, figures) — gitignored
data/                    # raw and processed data caches — gitignored
```

## Getting started

Requires Python 3.10+.

```bash
make setup       # creates .venv and installs requirements.txt
make app         # streamlit run app.py
```

Or without `make`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
streamlit run app.py
```

## Usage

**Streamlit app.** Set the universe, date range, forecast horizon, validation, and execution/risk parameters in the sidebar, then click **Run research pipeline**. The app is organised into seven tabs: Overview, Data & features, Time series, Models, Backtest, Risk, and Methodology.

**Headless pipeline:**

```bash
make run          # offline data, no hyperparameter tuning
make run-tuned    # offline data, 40 Optuna trials
```

or directly:

```bash
python scripts/run_pipeline.py --offline --n-trials 40
```

**Jupyter notebooks:**

```bash
make lab
```

**Tests:**

```bash
make test
```

**Clean generated output:**

```bash
make clean
```

## How it works

1. **Panel.** Daily bars are turned into a cross-sectionally standardised feature panel with vol-scaled labels and a volatility-regime tag.
2. **Validation.** Expanding-window walk-forward splits purge any training row whose label window overlaps the test block (plus an embargo), so overlapping labels can't leak. Imputation and scaling are fit on training rows only.
3. **Tuning.** Optuna (TPE, median pruning) tunes XGBoost to maximise mean rank IC divided by its cross-fold dispersion — ordering the cross-section is what a portfolio actually trades, not RMSE.
4. **Book construction.** Scores are smoothed, ranked per date, demeaned, truncated to the long/short tails, normalised to unit gross, held between rebalances, and scaled toward a target volatility using a trailing-vol leverage estimate.
5. **Costs.** Trades pay a half-spread plus a square-root market-impact term on realised turnover (Almgren-Chriss style).
6. **Honesty checks.** Deflated Sharpe Ratio, cost sensitivity, factor attribution, and a simulated-data null test guard against overfitting and leakage.

See the **Methodology** tab in the app for full detail on the target, features, validation, tuning, and portfolio construction, plus known limitations.

## Known limitations

- Daily bars only — no microstructure, no intraday fills, no queue position.
- The universe is fixed as of today, so survivorship bias isn't fully removed (point-in-time membership is a planned improvement).
- Borrow cost is a flat annual charge on short exposure rather than a per-name rate.
- The market-impact model is calibrated by assumption, not by fills, so capacity is untested.

## Requirements

Core dependencies: numpy, pandas, scipy, scikit-learn, statsmodels, pyarrow, xgboost, catboost, optuna, arch, hmmlearn, streamlit, plotly, matplotlib, seaborn, shap, and optionally yfinance for live prices. See `requirements.txt` for exact versions.
