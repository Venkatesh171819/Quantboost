"""Walk-forward training loop, backtest engine and the performance scorecard."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .costs import total_cost
from .labeling import sample_weights
from .models import fit_with_weights, make_regressors, rank_average
from .portfolio import rank_long_short_weights, vol_target_leverage
from .cv import walk_forward_splits
from .tuning import deflated_sharpe, fold_prep, rank_ic


def run_walk_forward(panel: pd.DataFrame, feats: list[str], xgb_params: dict | None = None,
                     n_folds: int = 5, horizon: int = 5, embargo: int = 5,
                     seed: int = 7, use_weights: bool = True, verbose: bool = True):
    """Train every model on each expanding window, predict the purged test block.

    Returns (oof_predictions, mean_feature_importance, fold_summary).
    """
    splits = walk_forward_splits(panel["date"], n_folds, horizon, embargo)
    blocks, importances, summary = [], {}, []

    for k, (tr_d, te_d) in enumerate(splits):
        tr = panel[panel["date"].isin(tr_d)]
        te = panel[panel["date"].isin(te_d)]
        xtr, xte = fold_prep(tr, te, feats)
        w = sample_weights(None, len(tr)) if use_weights else None

        block = te[["date", "ticker", "target", "fwd_ret"]].reset_index(drop=True).copy()
        block["fold"] = k + 1
        for name, model in make_regressors(xgb_params, seed=seed).items():
            fit_with_weights(model, xtr, tr["target"], w)
            block[name] = model.predict(xte)
            if hasattr(model, "feature_importances_"):
                importances.setdefault(name, []).append(
                    pd.Series(model.feature_importances_, index=feats))

        members = [c for c in block.columns
                   if c not in ("date", "ticker", "target", "fwd_ret", "fold")]
        block["Ensemble"] = rank_average(block[members])
        blocks.append(block)
        summary.append({"fold": k + 1, "train_days": len(tr_d), "test_days": len(te_d),
                        "train_end": str(pd.Timestamp(tr_d[-1]).date()),
                        "test_start": str(pd.Timestamp(te_d[0]).date()),
                        "test_end": str(pd.Timestamp(te_d[-1]).date())})
        if verbose:
            print(f"fold {k + 1}/{len(splits)}  train {len(tr_d)}d  "
                  f"test {len(te_d)}d  ({summary[-1]['test_start']} -> "
                  f"{summary[-1]['test_end']})")

    oof = pd.concat(blocks, ignore_index=True)
    imp = {k: pd.concat(v, axis=1).mean(axis=1).sort_values(ascending=False)
           for k, v in importances.items()}
    return oof, imp, pd.DataFrame(summary).set_index("fold")


def backtest(oof: pd.DataFrame, panel: pd.DataFrame, model: str, quantile: float = 0.2,
             half_spread_bps: float = 5.0, impact_bps: float = 8.0,
             vol_target: float = 0.10, borrow_bps: float = 50.0,
             rebalance: int | None = None, smooth: int = 3) -> pd.DataFrame:
    """Rank long/short book, vol targeted, charged spread + square-root impact.

    Positions formed on day t are traded into and earn day t+1 returns: the shift
    is what stops the backtest from trading on information it does not have.

    `rebalance` defaults to the label horizon: with a 5-day forecast you have no
    business trading every day, and daily reshuffling of a rank book generates
    turnover that costs more than the signal is worth. `smooth` applies a short
    EWMA to the scores first, which cuts turnover with almost no IC loss.
    """
    scores = oof.pivot_table(index="date", columns="ticker", values=model)
    fwd = panel.pivot_table(index="date", columns="ticker", values="ret1")
    fwd = fwd.reindex(index=scores.index, columns=scores.columns)

    if smooth and smooth > 1:
        scores = scores.ewm(span=smooth, min_periods=1).mean()

    w = rank_long_short_weights(scores, quantile)
    if rebalance is None:
        rebalance = 5
    if rebalance > 1:                      # hold the book between rebalance dates
        keep = np.zeros(len(w), dtype=bool)
        keep[::rebalance] = True
        w = w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0.0)
    gross_ret = (w.shift(1) * fwd).sum(axis=1)
    lev = vol_target_leverage(gross_ret, vol_target)
    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0) * lev
    shorts = w.clip(upper=0).abs().sum(axis=1) * lev
    cost = total_cost(turnover, shorts, half_spread_bps, impact_bps, borrow_bps)

    out = pd.DataFrame({"gross": lev * gross_ret, "cost": cost,
                        "turnover": turnover, "leverage": lev,
                        "gross_exposure": w.abs().sum(axis=1) * lev,
                        "net_exposure": w.sum(axis=1) * lev})
    out["net"] = out["gross"] - out["cost"]
    return out.dropna(subset=["net"])


def performance(bt: pd.DataFrame, ic: pd.Series | None = None,
                n_trials: int = 1) -> dict:
    """The scorecard. Note the Deflated Sharpe sitting next to the raw Sharpe."""
    r = bt["net"].dropna()
    ann_r, ann_v = r.mean() * 252, r.std() * math.sqrt(252)
    sharpe = ann_r / (ann_v + 1e-12)
    eq = (1 + r).cumprod()
    dd_series = eq / eq.cummax() - 1
    dd = float(dd_series.min())
    dd_dur = int((dd_series < 0).astype(int).groupby(
        (dd_series >= 0).cumsum()).sum().max() or 0)
    dn = r[r < 0].std() * math.sqrt(252)

    blocks = max(len(r) // 5, 1)
    fold_sr = r.groupby(np.arange(len(r)) // blocks).apply(
        lambda x: x.mean() / (x.std() + 1e-12))
    dsr = deflated_sharpe(float(r.mean() / (r.std() + 1e-12)), len(r), n_trials,
                          float(np.var(fold_sr, ddof=1)) if len(fold_sr) > 2 else 1e-4,
                          float(r.skew()), float(r.kurt() + 3))

    out = {
        "ann_return": float(ann_r), "ann_vol": float(ann_v), "sharpe_net": float(sharpe),
        "sharpe_gross": float(bt["gross"].mean() * 252 / (bt["gross"].std() * math.sqrt(252) + 1e-12)),
        "sortino": float(ann_r / (dn + 1e-12)),
        "max_drawdown": dd, "drawdown_days": dd_dur,
        "calmar": float(ann_r / (abs(dd) + 1e-12)),
        "hit_rate": float((r > 0).mean()),
        "tail_ratio": float(abs(np.quantile(r, 0.95) / (np.quantile(r, 0.05) + 1e-12))),
        "skew": float(r.skew()), "excess_kurtosis": float(r.kurt()),
        "t_stat": float(r.mean() / (r.std() / math.sqrt(len(r)) + 1e-12)),
        "ann_turnover": float(bt["turnover"].mean() * 252),
        "ann_cost_drag": float(bt["cost"].mean() * 252),
        "avg_leverage": float(bt["leverage"].mean()),
        "deflated_sharpe_prob": float(dsr),
    }
    if ic is not None and len(ic):
        out.update({"mean_ic": float(ic.mean()), "ic_ir": float(ic.mean() / (ic.std() + 1e-12)),
                    "ic_t_stat": float(ic.mean() / (ic.std() / math.sqrt(len(ic)) + 1e-12)),
                    "ic_hit_rate": float((ic > 0).mean())})
    return out


def equity_curve(bt: pd.DataFrame, col: str = "net") -> pd.Series:
    return (1 + bt[col].dropna()).cumprod()


def compare_models(oof: pd.DataFrame, panel: pd.DataFrame, models: list[str] | None = None,
                   n_trials: int = 1, **bt_kwargs) -> pd.DataFrame:
    models = models or [c for c in oof.columns
                        if c not in ("date", "ticker", "target", "fwd_ret", "fold")]
    rows = []
    for m in models:
        ic = rank_ic(oof.rename(columns={m: "pred"})[["date", "pred", "target"]])
        bt = backtest(oof, panel, m, **bt_kwargs)
        p = performance(bt, ic, n_trials)
        rows.append({"model": m, **{k: p[k] for k in
                                    ("mean_ic", "ic_ir", "ic_t_stat", "sharpe_net",
                                     "max_drawdown", "ann_turnover",
                                     "deflated_sharpe_prob") if k in p}})
    return pd.DataFrame(rows).set_index("model").sort_values("sharpe_net", ascending=False)


def cost_sensitivity(oof: pd.DataFrame, panel: pd.DataFrame, model: str,
                     multiples=(1, 2, 3, 5), **bt_kwargs) -> pd.DataFrame:
    """A strategy that only survives at 1x costs is not a strategy."""
    base_spread = bt_kwargs.pop("half_spread_bps", 5.0)
    base_impact = bt_kwargs.pop("impact_bps", 8.0)
    rows = []
    for m in multiples:
        bt = backtest(oof, panel, model, half_spread_bps=base_spread * m,
                      impact_bps=base_impact * m, **bt_kwargs)
        p = performance(bt)
        rows.append({"cost_multiple": f"{m}x", "sharpe_net": p["sharpe_net"],
                     "ann_return": p["ann_return"], "ann_cost_drag": p["ann_cost_drag"]})
    return pd.DataFrame(rows).set_index("cost_multiple")


def regime_breakdown(bt: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    reg = panel.groupby("date")["regime"].mean().round().reindex(bt.index).ffill()
    g = bt["net"].groupby(reg).agg(["mean", "std", "count"])
    g["sharpe"] = g["mean"] / (g["std"] + 1e-12) * math.sqrt(252)
    g.index = [["calm", "normal", "stressed"][int(i)] if np.isfinite(i) else "n/a"
               for i in g.index]
    return g


def ablation(panel: pd.DataFrame, blocks: dict[str, list[str]], **kwargs) -> pd.DataFrame:
    """Cumulative feature-block ablation: shows the marginal value of each block
    instead of one all-in number nobody can interrogate."""
    rows, used = [], []
    for name, cols in blocks.items():
        used += [c for c in cols if c in panel.columns]
        oof, _, _ = run_walk_forward(panel, used, verbose=False, **kwargs)
        ic = rank_ic(oof.rename(columns={"Ensemble": "pred"})[["date", "pred", "target"]])
        bt = backtest(oof, panel, "Ensemble")
        p = performance(bt, ic)
        rows.append({"block_added": name, "n_features": len(used),
                     "mean_ic": p["mean_ic"], "sharpe_net": p["sharpe_net"]})
    return pd.DataFrame(rows).set_index("block_added")
