"""Data layer: price panel loading with a fully offline simulated fallback.

The simulated generator is not decoration. A model that produces signal on
regime-switching GBM noise is leaking, so this doubles as a null test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA, SECTOR

try:                                     # optional
    import yfinance as yf
    HAS_YF = True
except Exception:                        # pragma: no cover
    HAS_YF = False

PANEL_COLS = ["date", "ticker", "close", "high", "low", "volume"]


def simulate_panel(tickers, start="2010-01-01", end="2024-12-31", seed=7) -> pd.DataFrame:
    """Two-state (calm/stressed) Markov vol regime + common market factor + idio noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    state, states = 0, np.zeros(n, dtype=int)
    for i in range(n):                                   # persistent regimes
        if rng.random() < (0.015 if state == 0 else 0.06):
            state = 1 - state
        states[i] = state
    mkt = rng.normal(0.0003, 1.0, n) * np.where(states == 0, 0.010, 0.026)

    frames = []
    for k, t in enumerate(tickers):
        beta = 0.5 + 1.2 * rng.random()
        ret = beta * mkt + rng.normal(0, 0.011 + 0.006 * rng.random(), n)
        close = 40 * (1 + k % 7) * np.exp(np.cumsum(ret))
        wick = np.abs(rng.normal(0, 0.008, n))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": t, "close": close,
            "high": close * (1 + wick), "low": close * (1 - wick),
            "volume": rng.lognormal(15.5, 0.45, n),
        }))
    panel = pd.concat(frames, ignore_index=True)
    panel.attrs["source"] = "Simulated (regime-switching GBM)"
    return panel


def download_panel(tickers, start, end) -> pd.DataFrame | None:
    """Adjusted daily bars from Yahoo. Returns None if unavailable."""
    if not HAS_YF:
        return None
    try:
        raw = yf.download(list(tickers), start=start, end=end,
                          auto_adjust=True, progress=False, threads=True)
    except Exception:
        return None
    if raw is None or len(raw) == 0:
        return None
    frames = []
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            df = (pd.DataFrame({"close": raw["Close"][t], "high": raw["High"][t],
                                "low": raw["Low"][t], "volume": raw["Volume"][t]})
                  if multi else
                  raw[["Close", "High", "Low", "Volume"]]
                  .rename(columns=str.lower).rename(columns={"close": "close"}))
        except KeyError:
            continue
        df = df.dropna()
        if len(df) < 400:                        # too short to validate on
            continue
        df["ticker"] = t
        frames.append(df.rename_axis("date").reset_index())
    if not frames:
        return None
    panel = pd.concat(frames, ignore_index=True)
    panel.attrs["source"] = "Yahoo Finance (adjusted)"
    return panel


def load_prices(tickers, start="2010-01-01", end="2024-12-31", offline=True,
                cache=True) -> pd.DataFrame:
    """Long panel: date, ticker, close, high, low, volume. Parquet-cached."""
    key = DATA / "raw" / f"panel_{len(tickers)}_{start}_{end}_{int(offline)}.parquet"
    if cache and key.exists():
        out = pd.read_parquet(key)
        out.attrs["source"] = "Local parquet cache"
        return out

    panel = None if offline else download_panel(tickers, start, end)
    if panel is None:
        panel = simulate_panel(tickers, start, end)

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)[PANEL_COLS]
    panel["sector"] = panel["ticker"].map(SECTOR).fillna("Other")
    if cache:
        key.parent.mkdir(parents=True, exist_ok=True)
        src = panel.attrs.get("source", "unknown")
        panel.to_parquet(key, index=False)
        panel.attrs["source"] = src
    return panel


def audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-name data quality: coverage, gaps, staleness, zero-volume days."""
    rows = []
    for t, g in panel.groupby("ticker"):
        g = g.sort_values("date")
        gaps = g["date"].diff().dt.days.fillna(1)
        rows.append({
            "ticker": t, "rows": len(g),
            "start": g["date"].min().date(), "end": g["date"].max().date(),
            "missing_close": int(g["close"].isna().sum()),
            "max_gap_days": int(gaps.max()),
            "stale_price_days": int((g["close"].diff() == 0).sum()),
            "zero_volume_days": int((g["volume"] <= 0).sum()),
        })
    return pd.DataFrame(rows).set_index("ticker").sort_index()
