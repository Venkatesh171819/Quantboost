"""Risk measurement: VaR, expected shortfall, and the backtests that decide
whether the VaR model itself is any good.

Breach clustering, not breach count, is what kills a risk model. Kupiec tests the
count; Christoffersen tests the independence.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(r: pd.Series, cl: float = 0.95) -> float:
    return float(-np.quantile(r.dropna(), 1 - cl))


def gaussian_var(r: pd.Series, cl: float = 0.95) -> float:
    r = r.dropna()
    return float(-(r.mean() + stats.norm.ppf(1 - cl) * r.std()))


def cornish_fisher_var(r: pd.Series, cl: float = 0.95) -> float:
    """Gaussian VaR corrected for skew and kurtosis. Cheap fix for fat tails."""
    r = r.dropna()
    z = stats.norm.ppf(1 - cl)
    s, k = r.skew(), r.kurt()
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
    return float(-(r.mean() + zcf * r.std()))


def expected_shortfall(r: pd.Series, cl: float = 0.95) -> float:
    r = r.dropna()
    cut = np.quantile(r, 1 - cl)
    tail = r[r <= cut]
    return float(-tail.mean()) if len(tail) else float("nan")


def filtered_historical_var(r: pd.Series, cl: float = 0.95, halflife: int = 20) -> float:
    """Devolatilise, take the empirical quantile, then revolatilise at today's vol.
    Keeps the fat tails but conditions on the current regime."""
    r = r.dropna()
    vol = r.ewm(halflife=halflife).std()
    std_res = (r / (vol + 1e-12)).dropna()
    q = np.quantile(std_res, 1 - cl)
    return float(-q * vol.iloc[-1])


def kupiec_pof(breaches: int, n: int, cl: float = 0.95) -> dict:
    """Proportion-of-failures likelihood ratio. Critical value 3.84 at 95%."""
    a = 1 - cl
    if n == 0 or breaches in (0, n):
        return {"lr": 0.0, "p_value": 1.0, "reject": False}
    p = breaches / n
    lr = -2 * (breaches * math.log(a) + (n - breaches) * math.log(1 - a)) + \
         2 * (breaches * math.log(p) + (n - breaches) * math.log(1 - p))
    return {"lr": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, 1)),
            "reject": bool(lr > 3.84)}


def christoffersen_independence(breach_flags: pd.Series) -> dict:
    """Are breaches clustered? Clustered breaches mean the model misses regime shifts."""
    x = breach_flags.astype(int).to_numpy()
    n00 = int(((x[:-1] == 0) & (x[1:] == 0)).sum())
    n01 = int(((x[:-1] == 0) & (x[1:] == 1)).sum())
    n10 = int(((x[:-1] == 1) & (x[1:] == 0)).sum())
    n11 = int(((x[:-1] == 1) & (x[1:] == 1)).sum())
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if min(p01, p11, p) <= 0 or max(p01, p11, p) >= 1:
        return {"lr": 0.0, "p_value": 1.0, "reject": False}
    ll_null = (n00 + n10) * math.log(1 - p) + (n01 + n11) * math.log(p)
    ll_alt = n00 * math.log(1 - p01) + n01 * math.log(p01) + \
             n10 * math.log(1 - p11) + n11 * math.log(p11)
    lr = -2 * (ll_null - ll_alt)
    return {"lr": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, 1)),
            "reject": bool(lr > 3.84)}


def var_table(r: pd.Series, levels=(0.95, 0.99)) -> pd.DataFrame:
    rows = []
    for cl in levels:
        hv = historical_var(r, cl)
        flags = (r < -hv)
        kup = kupiec_pof(int(flags.sum()), len(r), cl)
        chr_ = christoffersen_independence(flags)
        rows.append({
            "confidence": f"{cl:.0%}",
            "historical_var": hv,
            "gaussian_var": gaussian_var(r, cl),
            "cornish_fisher_var": cornish_fisher_var(r, cl),
            "filtered_hist_var": filtered_historical_var(r, cl),
            "expected_shortfall": expected_shortfall(r, cl),
            "breaches": int(flags.sum()),
            "expected_breaches": round((1 - cl) * len(r), 1),
            "kupiec_lr": round(kup["lr"], 2),
            "kupiec_reject": kup["reject"],
            "christoffersen_lr": round(chr_["lr"], 2),
            "christoffersen_reject": chr_["reject"],
        })
    return pd.DataFrame(rows).set_index("confidence")


def monte_carlo_paths(s0: float, mu: float, sigma: float, days: int = 252,
                      n_paths: int = 2000, seed: int = 7) -> np.ndarray:
    """Geometric Brownian motion under Ito's lemma: dS = mu*S*dt + sigma*S*dW.
    Used for stress testing and as a null-hypothesis data generator."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rng.standard_normal((n_paths, days))
    return s0 * np.exp(np.cumsum(shocks, axis=1))


def stress_scenarios(r: pd.Series) -> pd.DataFrame:
    """Deterministic shocks applied to the realised distribution."""
    rows = []
    for name, fn in {
        "base": lambda x: x,
        "vol x2": lambda x: x * 2,
        "worst 5 days repeated": lambda x: pd.concat([x, x.nsmallest(5)]),
        "fat left tail (-3 sigma shock)": lambda x: pd.concat(
            [x, pd.Series([-3 * x.std()])]),
    }.items():
        s = fn(r.dropna())
        rows.append({"scenario": name, "ann_vol": s.std() * math.sqrt(252),
                     "var_95": historical_var(s), "es_95": expected_shortfall(s)})
    return pd.DataFrame(rows).set_index("scenario")
