"""Turning forecasts into positions: covariance shrinkage, mean-variance, HRP,
Kelly sizing and volatility targeting.

A forecast is not a portfolio. This module is where the mathematical finance from
the MIT course actually earns its keep.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Shrink the sample covariance toward a scaled identity. Sample covariance on
    a short window is nearly singular and mean-variance will happily lever up its
    estimation error."""
    X = returns.dropna(how="any")
    n, p = X.shape
    S = np.cov(X.to_numpy(), rowvar=False)
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = np.linalg.norm(S - F, "fro") ** 2
    Xc = X.to_numpy() - X.to_numpy().mean(0)
    beta2 = sum(np.linalg.norm(np.outer(Xc[i], Xc[i]) - S, "fro") ** 2
                for i in range(n)) / n ** 2
    shrink = float(np.clip(beta2 / (d2 + 1e-12), 0.0, 1.0))
    return pd.DataFrame(shrink * F + (1 - shrink) * S, index=X.columns, columns=X.columns)


def mean_variance_weights(mu: pd.Series, cov: pd.DataFrame, risk_aversion: float = 5.0,
                          gross_limit: float = 1.0, name_limit: float = 0.1,
                          market_neutral: bool = True) -> pd.Series:
    """Closed-form Markowitz w = (1/lambda) * Sigma^-1 * mu, then constrained."""
    cols = [c for c in mu.index if c in cov.index]
    mu, cov = mu[cols], cov.loc[cols, cols]
    inv = np.linalg.pinv(cov.to_numpy() + 1e-8 * np.eye(len(cols)))
    w = pd.Series(inv @ mu.to_numpy() / risk_aversion, index=cols)
    if market_neutral:
        w = w - w.mean()
    w = w.clip(-name_limit, name_limit)
    gross = w.abs().sum()
    return w * (gross_limit / gross) if gross > 0 else w


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 25) -> pd.DataFrame:
    """Unconstrained frontier, for the classic risk/return picture."""
    inv = np.linalg.pinv(cov.to_numpy())
    one = np.ones(len(mu))
    m = mu.to_numpy()
    A, B, C = one @ inv @ m, m @ inv @ m, one @ inv @ one
    targets = np.linspace(m.min(), m.max(), n_points)
    rows = []
    for t in targets:
        d = B * C - A ** 2
        lam = (C * t - A) / (d + 1e-12)
        gam = (B - A * t) / (d + 1e-12)
        w = inv @ (lam * m + gam * one)
        rows.append({"target_return": t, "vol": math.sqrt(max(w @ cov.to_numpy() @ w, 0))})
    return pd.DataFrame(rows)


def hierarchical_risk_parity(cov: pd.DataFrame) -> pd.Series:
    """HRP: allocation by recursive bisection of a clustered covariance. No matrix
    inversion, so it is far more stable than mean-variance out of sample."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    corr = cov.div(np.sqrt(np.outer(np.diag(cov), np.diag(cov))) + 1e-12)
    dist = squareform(np.clip(np.sqrt(np.clip((1 - corr) / 2, 0, 1)), 0, 1), checks=False)
    order = [cov.index[i] for i in leaves_list(linkage(dist, "single"))]

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mid = len(c) // 2
            a, b = c[:mid], c[mid:]
            va = _cluster_var(cov, a)
            vb = _cluster_var(cov, b)
            alpha = 1 - va / (va + vb + 1e-12)
            w[a] *= alpha
            w[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items]
    ivp = 1 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub.to_numpy() @ ivp)


def rank_long_short_weights(scores: pd.DataFrame, quantile: float = 0.2,
                            name_limit: float = 0.15) -> pd.DataFrame:
    """Dollar-neutral, unit-gross weights from cross-sectional ranks.

    Ranks rather than raw forecasts, because only the ordering is trustworthy: a
    boosted tree is a good ranker and a badly calibrated point estimator.

    Discrete ranks make the two tails uneven whenever the cross-section does not
    divide cleanly, so neutrality is re-imposed on the selected names after
    clipping rather than assumed.
    """
    r = scores.rank(axis=1, pct=True)
    mask = (r <= quantile) | (r >= 1 - quantile)
    w = r.sub(r.mean(axis=1), axis=0).where(mask, 0.0).clip(-name_limit, name_limit)
    n_sel = mask.sum(axis=1).replace(0, np.nan)
    w = w - mask.mul(w.sum(axis=1) / n_sel, axis=0)          # exact dollar neutrality
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def kelly_fraction(mean: float, variance: float, fraction: float = 0.5,
                   cap: float = 3.0) -> float:
    """Fractional Kelly. Full Kelly maximises log wealth but assumes you know the
    moments exactly; with estimated moments, half Kelly is the honest choice."""
    if variance <= 0:
        return 0.0
    return float(np.clip(fraction * mean / variance, 0.0, cap))


def vol_target_leverage(returns: pd.Series, target: float = 0.10, window: int = 63,
                        cap: float = 3.0) -> pd.Series:
    """Scale exposure so realised vol tracks the target. Shifted by one day, since
    today's leverage can only use yesterday's estimate."""
    realised = returns.rolling(window).std() * math.sqrt(252)
    return (target / (realised + 1e-9)).shift(1).clip(0, cap).fillna(1.0)


def factor_attribution(strategy: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """OLS of strategy returns on factor returns with Newey-West style t-stats.

    This answers the question every buy-side interviewer asks: is the alpha real,
    or is it repackaged market beta, size and momentum?
    """
    df = pd.concat([strategy.rename("y"), factors], axis=1).dropna()
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    var = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
    se = np.sqrt(np.diag(var))
    names = ["alpha"] + list(df.drop(columns="y").columns)
    out = pd.DataFrame({"coef": beta, "std_err": se, "t_stat": beta / (se + 1e-12)},
                       index=names)
    out.loc["alpha", "annualised"] = beta[0] * 252
    return out
