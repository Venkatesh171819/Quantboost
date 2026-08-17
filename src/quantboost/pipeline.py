"""One-call orchestration: data -> features -> tuning -> walk-forward -> backtest.

Used by scripts/run_pipeline.py, by the notebooks and by the Streamlit app, so
there is exactly one implementation of the research process.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .backtest import (backtest, compare_models, cost_sensitivity, performance,
                       regime_breakdown, run_walk_forward)
from .config import CFG, Config
from .data import load_prices
from .features import build_panel, feature_columns
from .risk import var_table
from .tuning import rank_ic, tune_xgb


@dataclass
class Result:
    cfg: Config
    prices: pd.DataFrame
    panel: pd.DataFrame
    features: list[str]
    params: dict
    oof: pd.DataFrame
    importance: dict
    folds: pd.DataFrame
    source: str = ""
    study: object | None = None
    extras: dict = field(default_factory=dict)

    # ---- convenience accessors ----
    def ic(self, model: str = "Ensemble") -> pd.Series:
        return rank_ic(self.oof.rename(columns={model: "pred"})[["date", "pred", "target"]])

    def book(self, model: str = "Ensemble") -> pd.DataFrame:
        return backtest(self.oof, self.panel, model, self.cfg.quantile,
                        self.cfg.half_spread_bps, self.cfg.impact_bps, self.cfg.vol_target)

    def scorecard(self, model: str = "Ensemble") -> pd.Series:
        p = performance(self.book(model), self.ic(model), max(self.cfg.n_trials, 1))
        return pd.Series(p, name=model)

    def leaderboard(self) -> pd.DataFrame:
        return compare_models(self.oof, self.panel, n_trials=max(self.cfg.n_trials, 1),
                             quantile=self.cfg.quantile,
                             half_spread_bps=self.cfg.half_spread_bps,
                             impact_bps=self.cfg.impact_bps,
                             vol_target=self.cfg.vol_target)

    def costs(self, model: str = "Ensemble") -> pd.DataFrame:
        return cost_sensitivity(self.oof, self.panel, model, quantile=self.cfg.quantile,
                                half_spread_bps=self.cfg.half_spread_bps,
                                impact_bps=self.cfg.impact_bps,
                                vol_target=self.cfg.vol_target)

    def risk(self, model: str = "Ensemble") -> pd.DataFrame:
        return var_table(self.book(model)["net"])

    def regimes(self, model: str = "Ensemble") -> pd.DataFrame:
        return regime_breakdown(self.book(model), self.panel)


def run(cfg: Config | None = None, verbose: bool = True) -> Result:
    cfg = cfg or CFG
    if verbose:
        print(f"[1/4] loading prices  ({len(cfg.universe)} names, "
              f"{'simulated' if cfg.offline else 'Yahoo'})")
    prices = load_prices(cfg.universe, cfg.start, cfg.end, cfg.offline)
    source = prices.attrs.get("source", "unknown")

    if verbose:
        print("[2/4] building feature panel")
    panel = build_panel(prices, cfg.horizon)
    feats = [c for c in feature_columns() if c in panel.columns]

    if verbose:
        print(f"[3/4] tuning ({cfg.n_trials} trials)" if cfg.n_trials else "[3/4] tuning skipped")
    params, study = tune_xgb(panel, feats, cfg.n_trials, cfg.horizon, cfg.embargo, cfg.seed)

    if verbose:
        print("[4/4] purged walk-forward")
    oof, imp, folds = run_walk_forward(panel, feats, params, cfg.n_folds, cfg.horizon,
                                       cfg.embargo, cfg.seed, verbose=verbose)

    res = Result(cfg=cfg, prices=prices, panel=panel, features=feats, params=params,
                 oof=oof, importance=imp, folds=folds, source=source, study=study)
    if verbose:
        print("\nEnsemble scorecard")
        print(res.scorecard().round(4).to_string())
    return res
