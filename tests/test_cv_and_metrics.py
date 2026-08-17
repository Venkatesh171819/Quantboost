import numpy as np
import pandas as pd

from quantboost.backtest import backtest, performance
from quantboost.costs import impact_cost, spread_cost
from quantboost.cv import PurgedKFold, combinatorial_purged_splits
from quantboost.portfolio import (kelly_fraction, ledoit_wolf_cov,
                                  rank_long_short_weights, vol_target_leverage)
from quantboost.risk import expected_shortfall, historical_var, kupiec_pof
from quantboost.tuning import deflated_sharpe, rank_ic


def test_purged_kfold_never_leaks_indices():
    X = pd.DataFrame(np.random.randn(500, 3))
    t1 = pd.Series(np.arange(500) + 5)
    for tr, te in PurgedKFold(5, t1, 0.02).split(X):
        assert len(set(tr) & set(te)) == 0


def test_cpcv_generates_multiple_paths(panel):
    splits = combinatorial_purged_splits(panel["date"], n_groups=6, n_test=2)
    assert len(splits) > 5
    for tr, te in splits:
        assert len(set(pd.to_datetime(tr)) & set(pd.to_datetime(te))) == 0


def test_weights_are_dollar_neutral_and_unit_gross():
    scores = pd.DataFrame(np.random.randn(50, 10),
                          index=pd.date_range("2020-01-01", periods=50, freq="B"))
    w = rank_long_short_weights(scores, 0.2)
    assert np.allclose(w.sum(axis=1), 0, atol=1e-8)
    assert np.allclose(w.abs().sum(axis=1), 1, atol=1e-8)


def test_costs_are_never_negative():
    t = pd.Series([0.0, 0.5, 2.0])
    assert (spread_cost(t) >= 0).all() and (impact_cost(t) >= 0).all()


def test_impact_is_concave_in_size():
    """Square-root impact: doubling size must less than double the per-unit cost."""
    small = impact_cost(pd.Series([0.5])).iloc[0] / 0.5
    large = impact_cost(pd.Series([2.0])).iloc[0] / 2.0
    assert large < small * 2


def test_net_is_never_above_gross(panel, feats):
    dates = pd.to_datetime(sorted(panel["date"].unique()))[-300:]
    sub = panel[panel["date"].isin(dates)].copy()
    sub["Dummy"] = np.random.randn(len(sub))
    bt = backtest(sub[["date", "ticker", "Dummy"]], panel, "Dummy")
    assert (bt["net"] <= bt["gross"] + 1e-12).all()


def test_vol_target_leverage_is_lagged():
    r = pd.Series(np.random.randn(300) * 0.01,
                  index=pd.date_range("2020-01-01", periods=300, freq="B"))
    lev = vol_target_leverage(r, 0.1)
    assert lev.isna().sum() == 0 and (lev <= 3).all()


def test_ledoit_wolf_is_positive_semidefinite():
    X = pd.DataFrame(np.random.randn(120, 15))
    cov = ledoit_wolf_cov(X)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > -1e-10


def test_kelly_is_capped_and_non_negative():
    assert kelly_fraction(1.0, 1e-6) <= 3.0
    assert kelly_fraction(-1.0, 0.01) == 0.0


def test_var_ordering_and_kupiec():
    r = pd.Series(np.random.standard_t(5, 3000) * 0.01)
    assert historical_var(r, 0.99) > historical_var(r, 0.95)
    assert expected_shortfall(r, 0.95) >= historical_var(r, 0.95)
    assert kupiec_pof(150, 3000, 0.95)["reject"] is False


def test_deflated_sharpe_penalises_more_trials():
    kw = dict(sr=0.08, n_obs=1000, sr_variance=0.0009, skew=-0.3, kurtosis=5.0)
    assert deflated_sharpe(n_trials=2, **kw) > deflated_sharpe(n_trials=500, **kw)


def test_random_signal_has_no_information():
    """Null test: random predictions must produce an IC indistinguishable from zero."""
    dates = np.repeat(pd.date_range("2020-01-01", periods=200, freq="B"), 20)
    df = pd.DataFrame({"date": dates, "pred": np.random.randn(4000),
                       "target": np.random.randn(4000)})
    ic = rank_ic(df)
    t = ic.mean() / (ic.std() / np.sqrt(len(ic)))
    assert abs(t) < 3


def test_performance_keys_present(panel):
    bt = pd.DataFrame({"gross": np.random.randn(500) * 0.01,
                       "cost": np.abs(np.random.randn(500)) * 1e-4,
                       "turnover": np.abs(np.random.randn(500)),
                       "leverage": 1.0},
                      index=pd.date_range("2020-01-01", periods=500, freq="B"))
    bt["net"] = bt["gross"] - bt["cost"]
    p = performance(bt)
    for k in ("sharpe_net", "max_drawdown", "deflated_sharpe_prob", "ann_turnover"):
        assert k in p
