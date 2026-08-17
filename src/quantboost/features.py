"""Feature factory.

Hard rule enforced everywhere in this module: a feature dated t may only use
information observable at the close of t. Anything forward-looking belongs in
`labeling.py`, never here.

Blocks
------
trend/momentum, mean reversion, technical oscillators, volatility, higher moments,
liquidity/microstructure proxies, memory (fractional differencing), calendar,
complexity/entropy, regime tags, then cross-sectional standardisation per date.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- primitives


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def macd_hist(s: pd.Series) -> pd.Series:
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    return (macd - macd.ewm(span=9, adjust=False).mean()) / (s.rolling(63).std() + 1e-9)


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def frac_diff_weights(d: float, width: int) -> np.ndarray:
    w = [1.0]
    for k in range(1, width):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])


def frac_diff(s: pd.Series, d: float = 0.4, width: int = 24) -> pd.Series:
    """Fixed-width fractional differencing: stationary but keeps long memory,
    unlike a naive first difference which throws the level information away."""
    w = frac_diff_weights(d, width)
    x = np.log(s.to_numpy(dtype=float))
    out = np.full(len(x), np.nan)
    for i in range(width, len(x)):
        out[i] = float(w @ x[i - width:i])
    return pd.Series(out, index=s.index)


def shannon_entropy(r: pd.Series, window: int = 60, bins: int = 8) -> pd.Series:
    def _h(x):
        cnt, _ = np.histogram(x, bins=bins)
        p = cnt / max(cnt.sum(), 1)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    return r.rolling(window).apply(_h, raw=True)


def hurst_rs(x: np.ndarray) -> float:
    """Rescaled-range Hurst exponent. >0.5 trending, <0.5 mean reverting."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 32 or np.allclose(x.std(), 0):
        return np.nan
    sizes = [s for s in (8, 16, 32, 64, 128, 256) if s <= n // 2]
    rs = []
    for s in sizes:
        vals = []
        for i in range(0, n - s + 1, s):
            seg = x[i:i + s]
            dev = np.cumsum(seg - seg.mean())
            sd = seg.std(ddof=1)
            if sd > 0:
                vals.append((dev.max() - dev.min()) / sd)
        if vals:
            rs.append(np.mean(vals))
    if len(rs) < 3:
        return np.nan
    return float(np.polyfit(np.log(sizes[:len(rs)]), np.log(rs), 1)[0])


def rolling_hurst(r: pd.Series, window: int = 126, step: int = 5) -> pd.Series:
    """Decimated rolling Hurst: computed every `step` bars, forward filled."""
    out = pd.Series(np.nan, index=r.index)
    x = r.to_numpy(dtype=float)
    for i in range(window, len(x), step):
        out.iloc[i] = hurst_rs(x[i - window:i])
    return out.ffill()


# --------------------------------------------------------------- per-ticker

RAW_FEATURES = [
    # trend / momentum
    "mom5", "mom21", "mom63", "mom252", "ma_ratio", "dist_52w_high",
    # reversal
    "rev1", "rev5", "bb_z",
    # oscillators
    "rsi14", "macd_h", "stoch_k",
    # volatility
    "vol20", "vol63", "vol_ratio", "ewma_vol", "vov", "dn_vol", "atr_pct", "parkinson",
    # higher moments
    "skew60", "kurt60",
    # liquidity / microstructure
    "vol_z", "amihud", "dollar_vol", "roll_spread", "hl_range",
    # memory / complexity
    "fdiff", "entropy60", "hurst126",
    # gaps
    "gap_z",
]

CATEGORICALS = ["regime", "sector", "dow", "month"]
EXTRA_NUMERIC = ["month_end", "quarter_end"]


def per_ticker_features(g: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """All single-name features plus the forward label. `g` is one ticker, date-sorted."""
    g = g.sort_values("date").reset_index(drop=True).copy()
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    r = np.log(c).diff()
    g["ret1"] = r

    # trend / momentum
    g["mom5"] = np.log(c / c.shift(5))
    g["mom21"] = np.log(c / c.shift(21))
    g["mom63"] = np.log(c / c.shift(63))
    g["mom252"] = np.log(c.shift(21) / c.shift(252))       # 12-1, skips reversal month
    g["ma_ratio"] = c / c.rolling(50).mean() - 1
    g["dist_52w_high"] = c / c.rolling(252).max() - 1

    # reversal
    g["rev1"] = -r
    g["rev5"] = -np.log(c / c.shift(5))
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    g["bb_z"] = (c - ma20) / (sd20 + 1e-9)

    # oscillators
    g["rsi14"] = rsi(c) / 100 - 0.5
    g["macd_h"] = macd_hist(c)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    g["stoch_k"] = (c - ll) / (hh - ll + 1e-9) - 0.5

    # volatility
    g["vol20"] = r.rolling(20).std()
    g["vol63"] = r.rolling(63).std()
    g["vol_ratio"] = g["vol20"] / (g["vol63"] + 1e-9) - 1
    g["ewma_vol"] = r.ewm(halflife=20).std()
    g["vov"] = g["vol20"].rolling(60).std()
    g["dn_vol"] = r.clip(upper=0).rolling(60).std()
    tr = true_range(h, l, c)
    g["atr_pct"] = tr.rolling(14).mean() / c
    g["parkinson"] = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * math.log(2)))

    # higher moments
    g["skew60"] = r.rolling(60).skew()
    g["kurt60"] = r.rolling(60).kurt()

    # liquidity / microstructure proxies
    lv = np.log1p(v)
    g["vol_z"] = (lv - lv.rolling(20).mean()) / (lv.rolling(20).std() + 1e-9)
    g["amihud"] = (r.abs() / (c * v + 1e-9)).rolling(20).mean() * 1e9
    g["dollar_vol"] = np.log1p((c * v).rolling(20).mean())
    cov = r.rolling(20).cov(r.shift(1))
    g["roll_spread"] = 2 * np.sqrt(np.maximum(-cov, 0))     # Roll (1984) estimator
    g["hl_range"] = (h - l) / c

    # memory / complexity
    g["fdiff"] = frac_diff(c)
    g["entropy60"] = shannon_entropy(r.fillna(0))
    # Hurst is expensive; evaluate every 5th bar and hold the last value between
    # evaluations. It is a slow-moving statistic, so the approximation is free.
    g["hurst126"] = rolling_hurst(r.fillna(0), window=126, step=5)

    # gap
    g["gap_z"] = (r - r.rolling(20).mean()) / (g["vol20"] + 1e-9)

    # volatility regime from an EXPANDING quantile: uses the past only
    q = g["vol20"].expanding(252).rank(pct=True)
    g["regime"] = pd.cut(q, [-0.01, 0.4, 0.75, 1.01], labels=[0, 1, 2]).astype(float)

    # ---- labels (the only forward-looking columns in the whole panel) ----
    fwd = np.log(c.shift(-horizon) / c)
    g["fwd_ret"] = fwd
    g["target"] = fwd / (g["vol20"] * math.sqrt(horizon) + 1e-9)   # vol-scaled
    g["target_bin"] = np.sign(fwd)
    # label end time, needed by purged CV to know what overlaps what
    g["t1"] = g["date"].shift(-horizon)
    return g


def cross_sectional_zscore(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Standardise per date, not per series. Turns level forecasting into
    relative value, which is what a market-neutral book actually trades."""
    grp = panel.groupby("date")[cols]
    z = (panel[cols] - grp.transform("mean")) / (grp.transform("std") + 1e-9)
    z.columns = [f"{c}_cs" for c in cols]
    return z.clip(-5, 5)


def build_panel(prices: pd.DataFrame, horizon: int = 5, clip_target: float = 6.0) -> pd.DataFrame:
    """Full feature panel, one row per (date, ticker)."""
    parts = [per_ticker_features(g, horizon) for _, g in prices.groupby("ticker", sort=False)]
    panel = pd.concat(parts, ignore_index=True)

    d = pd.to_datetime(panel["date"])
    panel["dow"] = d.dt.dayofweek
    panel["month"] = d.dt.month
    panel["month_end"] = d.dt.is_month_end.astype(int)
    panel["quarter_end"] = d.dt.is_quarter_end.astype(int)
    # cyclical encoding of seasonality, so December and January stay neighbours
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)

    panel = pd.concat([panel, cross_sectional_zscore(panel, RAW_FEATURES)], axis=1)
    panel["target"] = panel["target"].clip(-clip_target, clip_target)
    panel = panel.dropna(subset=["target", "vol20"]).reset_index(drop=True)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_columns(cross_sectional: bool = True, include_cat: bool = True) -> list[str]:
    base = [f"{c}_cs" for c in RAW_FEATURES] if cross_sectional else list(RAW_FEATURES)
    base += EXTRA_NUMERIC + ["month_sin", "month_cos"]
    if include_cat:
        base += ["regime", "dow"]
    return base
