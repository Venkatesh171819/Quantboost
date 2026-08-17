"""Time series analysis: diagnostics, volatility models, regimes, baselines.

Every function here produces either a diagnostic you report, a feature you feed
the model, or a baseline you must beat. Nothing is decorative.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA
    HAS_SM = True
except Exception:                                       # pragma: no cover
    HAS_SM = False

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except Exception:
    HAS_HMM = False

from .features import frac_diff, hurst_rs


# ------------------------------------------------------------- diagnostics

def stationarity_report(series: dict[str, pd.Series]) -> pd.DataFrame:
    """ADF (null: unit root) and KPSS (null: stationary). Read them together:
    agreement is informative, disagreement means fractional integration."""
    rows = []
    for name, s in series.items():
        s = pd.Series(s).dropna()
        row = {"series": name, "n": len(s)}
        if HAS_SM and len(s) > 50:
            row["adf_stat"], row["adf_p"] = adfuller(s, autolag="AIC")[:2]
            try:
                row["kpss_stat"], row["kpss_p"] = kpss(s, regression="c", nlags="auto")[:2]
            except Exception:
                row["kpss_stat"], row["kpss_p"] = np.nan, np.nan
        row["hurst"] = hurst_rs(s.to_numpy())
        row["verdict"] = ("stationary" if row.get("adf_p", 1) < 0.05 else "unit root")
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")


def min_frac_diff_order(price: pd.Series, grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9),
                        alpha: float = 0.05) -> pd.DataFrame:
    """Smallest d that passes ADF: the least memory you must destroy."""
    rows = []
    for d in grid:
        s = frac_diff(price, d=d).dropna()
        p = adfuller(s, autolag="AIC")[1] if (HAS_SM and len(s) > 50) else np.nan
        rows.append({"d": d, "adf_p": p, "passes": bool(p is not np.nan and p < alpha),
                     "corr_with_log_price": float(np.corrcoef(
                         s, np.log(price).reindex(s.index))[0, 1])})
    return pd.DataFrame(rows)


def autocorrelation_report(r: pd.Series, lags: int = 15) -> pd.DataFrame:
    r = r.dropna()
    if HAS_SM:
        a = acf(r, nlags=lags, fft=True)[1:]
        p = pacf(r, nlags=lags)[1:]
        a2 = acf(r ** 2, nlags=lags, fft=True)[1:]
        lb = acorr_ljungbox(r, lags=[lags], return_df=True)
        lb2 = acorr_ljungbox(r ** 2, lags=[lags], return_df=True)
        out = pd.DataFrame({"acf": a, "pacf": p, "acf_squared": a2},
                           index=range(1, lags + 1))
        out.attrs["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
        out.attrs["ljung_box_squared_p"] = float(lb2["lb_pvalue"].iloc[0])
        return out
    a = [r.autocorr(k) for k in range(1, lags + 1)]
    return pd.DataFrame({"acf": a}, index=range(1, lags + 1))


# ------------------------------------------------------------- volatility

def har_rv(r: pd.Series) -> pd.DataFrame:
    """HAR-RV features: daily, weekly, monthly realised vol. Cheap, and hard to beat."""
    rv = r.pow(2).rolling(1).sum()
    return pd.DataFrame({
        "rv_d": np.sqrt(rv), "rv_w": np.sqrt(rv.rolling(5).mean()),
        "rv_m": np.sqrt(rv.rolling(22).mean()),
    })


def fit_garch(r: pd.Series, p: int = 1, q: int = 1, dist: str = "t", model: str = "GARCH"):
    """GARCH / EGARCH / GJR conditional volatility. Falls back to EWMA if `arch`
    is not installed, so downstream code never breaks."""
    r = r.dropna() * 100
    if HAS_ARCH and len(r) > 250:
        try:
            spec = arch_model(r, vol="Garch" if model == "GARCH" else model,
                             p=p, o=1 if model == "GJR" else 0, q=q, dist=dist)
            res = spec.fit(disp="off", show_warning=False)
            cond = res.conditional_volatility / 100
            return cond, {"model": model, "aic": res.aic,
                          "persistence": float(sum(v for k, v in res.params.items()
                                                   if k.startswith(("alpha", "beta"))))}
        except Exception:
            pass
    cond = (r / 100).ewm(halflife=20).std()
    return cond, {"model": "EWMA fallback", "aic": np.nan, "persistence": np.nan}


# ------------------------------------------------------------- regimes

def hmm_regimes(r: pd.Series, n_states: int = 3, seed: int = 7) -> pd.Series:
    """Hidden-state labels ordered by volatility (0 = calm). Used as a categorical
    feature and as the conditioning variable for per-regime reporting."""
    r = r.dropna()
    if HAS_HMM and len(r) > 500:
        try:
            X = np.column_stack([r.to_numpy(), r.rolling(20).std().bfill().to_numpy()])
            m = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=200, random_state=seed).fit(X)
            states = pd.Series(m.predict(X), index=r.index)
            order = states.map(r.groupby(states).std().rank().astype(int) - 1)
            return order.astype(int)
        except Exception:
            pass
    q = r.rolling(20).std().expanding(252).rank(pct=True)
    return pd.cut(q, [-0.01, 0.4, 0.75, 1.01], labels=list(range(n_states))).astype(float)


def cusum_breaks(r: pd.Series, threshold: float = 5.0) -> list:
    """Symmetric CUSUM filter: dates where cumulative drift exceeds a vol threshold.
    Justifies how often you retrain."""
    s = (r - r.mean()) / (r.std() + 1e-12)
    pos = neg = 0.0
    events = []
    for t, x in s.dropna().items():
        pos, neg = max(0.0, pos + x), min(0.0, neg + x)
        if pos > threshold:
            pos = 0.0
            events.append(t)
        elif neg < -threshold:
            neg = 0.0
            events.append(t)
    return events


# ------------------------------------------------------------- baselines

def baseline_forecasts(r: pd.Series, horizon: int = 5, test_frac: float = 0.3) -> pd.DataFrame:
    """Random walk, historical mean, AR(1), ARIMA(1,0,1). If boosting cannot beat
    these out of sample, that is the headline finding, not a bug."""
    r = r.dropna()
    y = r.rolling(horizon).sum().shift(-horizon).dropna()
    split = int(len(y) * (1 - test_frac))
    tr, te = y.iloc[:split], y.iloc[split:]
    preds = {"random_walk": np.zeros(len(te)),
             "historical_mean": np.full(len(te), tr.mean())}
    if HAS_SM and len(tr) > 300:
        for name, order in (("ar1", (1, 0, 0)), ("arima_101", (1, 0, 1))):
            try:
                fit = ARIMA(tr, order=order).fit()
                preds[name] = fit.forecast(steps=len(te)).to_numpy()
            except Exception:
                preds[name] = np.full(len(te), tr.mean())
    rows = []
    for name, p in preds.items():
        err = te.to_numpy() - p
        rows.append({"model": name,
                     "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "mae": float(np.mean(np.abs(err))),
                     "directional_acc": float(np.mean(np.sign(p) == np.sign(te.to_numpy())))})
    return pd.DataFrame(rows).set_index("model")


def yield_curve_pcs(curve: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Level / slope / curvature from a term-structure panel via PCA."""
    X = curve.dropna()
    Xc = X - X.mean()
    u, s, vt = np.linalg.svd(Xc.to_numpy(), full_matrices=False)
    pcs = u[:, :n] * s[:n]
    return pd.DataFrame(pcs, index=X.index,
                        columns=["level", "slope", "curvature"][:n])


def engle_granger(a: pd.Series, b: pd.Series) -> dict:
    """Two-step cointegration test plus the OLS hedge ratio."""
    df = pd.concat([a, b], axis=1).dropna()
    x, y = df.iloc[:, 1].to_numpy(), df.iloc[:, 0].to_numpy()
    beta = float(np.polyfit(x, y, 1)[0])
    spread = pd.Series(y - beta * x, index=df.index)
    p = adfuller(spread, autolag="AIC")[1] if HAS_SM else np.nan
    hl = np.nan
    d = spread.diff().dropna()
    lag = spread.shift().dropna().reindex(d.index)
    if len(d) > 30:
        k = float(np.polyfit(lag, d, 1)[0])
        if k < 0:
            hl = float(-math.log(2) / k)
    return {"hedge_ratio": beta, "adf_p": p, "half_life_days": hl,
            "cointegrated": bool(p == p and p < 0.05)}


def kalman_hedge_ratio(y: pd.Series, x: pd.Series, q: float = 1e-5,
                       r_obs: float = 1e-3) -> pd.Series:
    """Time-varying hedge ratio via a scalar Kalman filter (random-walk beta)."""
    df = pd.concat([y, x], axis=1).dropna()
    beta, P = 0.0, 1.0
    out = []
    for yy, xx in df.to_numpy():
        P += q
        k = P * xx / (xx * P * xx + r_obs)
        beta += k * (yy - xx * beta)
        P *= (1 - k * xx)
        out.append(beta)
    return pd.Series(out, index=df.index, name="kalman_beta")
