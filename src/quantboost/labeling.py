"""Labels and sample weights (Lopez de Prado, Advances in Financial ML).

Why this exists: fixed-horizon labels ignore the path, so a trade that would have
been stopped out still counts as a win. Triple-barrier labels fix that, and
overlapping label windows break the iid assumption, which sample weights fix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(close: pd.Series, vol: pd.Series, pt: float = 2.0, sl: float = 1.0,
                   max_hold: int = 10) -> pd.DataFrame:
    """Which barrier is touched first: profit-take, stop-loss, or the time limit.

    Barriers are multiples of conditional volatility, so they adapt to regime
    instead of being a fixed percentage that is loose in 2008 and tight in 2017.
    """
    close = close.astype(float)
    idx = close.index
    label = np.zeros(len(idx))
    held = np.full(len(idx), np.nan)
    ret = np.full(len(idx), np.nan)
    cv = vol.reindex(idx).to_numpy()
    px = close.to_numpy()

    for i in range(len(idx) - 1):
        s = cv[i]
        if not np.isfinite(s) or s <= 0:
            continue
        path = np.log(px[i + 1:i + 1 + max_hold] / px[i])
        if path.size == 0:
            continue
        up = np.flatnonzero(path > pt * s)
        dn = np.flatnonzero(path < -sl * s)
        first_up = up[0] if up.size else np.inf
        first_dn = dn[0] if dn.size else np.inf
        if np.isinf(first_up) and np.isinf(first_dn):
            label[i], held[i], ret[i] = 0.0, path.size, path[-1]
        else:
            j = int(min(first_up, first_dn))
            label[i] = 1.0 if first_up < first_dn else -1.0
            held[i], ret[i] = j + 1, path[j]
    return pd.DataFrame({"label": label, "bars_held": held, "realised": ret}, index=idx)


def label_end_times(dates: pd.Series, horizon: int) -> pd.Series:
    """t1: when the label of the observation at t is finally known."""
    d = pd.Series(pd.to_datetime(dates).to_numpy(), index=pd.to_datetime(dates).to_numpy())
    return d.shift(-horizon).ffill()


def average_uniqueness(t1: pd.Series) -> pd.Series:
    """How much of each label's window is not shared with other labels.

    Overlapping windows mean a single market move is counted many times; weighting
    by uniqueness stops the model from over-learning crowded periods.
    """
    t1 = t1.dropna().sort_index()
    bars = t1.index.union(t1.to_numpy())
    count = pd.Series(0.0, index=bars)
    for t0, tend in t1.items():
        count.loc[t0:tend] += 1.0
    out = {}
    for t0, tend in t1.items():
        seg = count.loc[t0:tend]
        out[t0] = float((1.0 / seg.replace(0, np.nan)).mean())
    return pd.Series(out).reindex(t1.index).fillna(1.0)


def time_decay_weights(n: int, last_weight: float = 1.0, first_weight: float = 0.35):
    """Linear recency decay. Old regimes still inform, they just count for less."""
    return np.linspace(first_weight, last_weight, max(n, 1))


def sample_weights(t1: pd.Series | None, n: int, decay: bool = True) -> np.ndarray:
    w = np.ones(n)
    if t1 is not None and len(t1.dropna()) == n:
        try:
            u = average_uniqueness(t1).to_numpy()
            if len(u) == n:
                w = u / (u.mean() + 1e-12)
        except Exception:
            pass
    if decay:
        w = w * time_decay_weights(n)
    return w / (w.mean() + 1e-12)


def meta_labels(primary_side: pd.Series, realised: pd.Series) -> pd.Series:
    """1 if taking the primary model's bet made money, else 0.

    The secondary model then learns *when to trust the primary*, which lifts
    precision and gives you a natural position-sizing probability.
    """
    return ((np.sign(primary_side) * np.sign(realised)) > 0).astype(int)
