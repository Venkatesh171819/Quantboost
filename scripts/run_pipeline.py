#!/usr/bin/env python3
"""Headless run: data -> features -> tuning -> walk-forward -> backtest -> reports.

    python scripts/run_pipeline.py --offline --n-trials 0 --folds 5

Writes CSV artefacts and PNG figures into reports/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from quantboost.config import Config, DEFAULT_UNIVERSE
from quantboost.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the QuantBoost research pipeline")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 5, 21])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=5)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--vol-target", type=float, default=0.10)
    ap.add_argument("--offline", action="store_true", help="use the simulated panel")
    ap.add_argument("--live", dest="offline", action="store_false")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_UNIVERSE)
    ap.set_defaults(offline=True)
    a = ap.parse_args()

    cfg = Config(universe=list(a.tickers), start=a.start, end=a.end, offline=a.offline,
                 horizon=a.horizon, quantile=a.quantile, n_folds=a.folds,
                 embargo=a.embargo, n_trials=a.n_trials, vol_target=a.vol_target)
    res = run(cfg)

    out = ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    res.leaderboard().to_csv(out / "model_leaderboard.csv")
    res.scorecard().to_csv(out / "scorecard.csv")
    res.costs().to_csv(out / "cost_sensitivity.csv")
    res.risk().to_csv(out / "var_backtests.csv")
    res.regimes().to_csv(out / "regime_breakdown.csv")
    res.folds.to_csv(out / "folds.csv")
    res.oof.to_parquet(out / "oof_predictions.parquet", index=False)
    pd.Series(res.params).to_csv(out / "tuned_params.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from quantboost.backtest import equity_curve
        from quantboost.plotting import plot_drawdown, plot_equity, plot_ic
        bt = res.book()
        plot_equity({"Ensemble, net": equity_curve(bt),
                     "Ensemble, gross": equity_curve(bt, "gross")})
        plt.savefig(out / "figures" / "equity.png", bbox_inches="tight")
        plt.close()
        plot_drawdown(equity_curve(bt))
        plt.savefig(out / "figures" / "drawdown.png", bbox_inches="tight")
        plt.close()
        plot_ic(res.ic())
        plt.savefig(out / "figures" / "ic.png", bbox_inches="tight")
        plt.close()
    except Exception as exc:                      # pragma: no cover
        print(f"figures skipped: {exc}")

    print(f"\nartefacts written to {out}")


if __name__ == "__main__":
    main()
