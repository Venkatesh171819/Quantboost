"""The leakage suite. This is the most important file in the repository.

A backtest is a claim about the future. These tests are the only reason anyone
should believe the claim.
"""
import numpy as np
import pandas as pd
import pytest

from quantboost.cv import assert_no_overlap, walk_forward_splits
from quantboost.features import RAW_FEATURES, per_ticker_features
from quantboost.tuning import fold_prep


def test_features_do_not_use_the_future(prices):
    """Truncating the series must not change features already computed.

    If a feature at time t depends on data after t, the truncated version will
    differ. This catches accidental centred windows and negative shifts.
    """
    g = prices[prices["ticker"] == "AAA"].reset_index(drop=True)
    full = per_ticker_features(g, horizon=5).set_index("date")
    trunc = per_ticker_features(g.iloc[:-60].copy(), horizon=5).set_index("date")
    common = trunc.index[-200:]
    for col in RAW_FEATURES:
        a = full.loc[common, col]
        b = trunc.loc[common, col]
        both = a.notna() & b.notna()
        assert np.allclose(a[both], b[both], atol=1e-8, rtol=1e-6), f"{col} looks forward"


def test_target_is_strictly_forward(prices):
    """The label at t must equal the realised return from t to t+h. No off-by-one."""
    g = prices[prices["ticker"] == "BBB"].reset_index(drop=True)
    f = per_ticker_features(g, horizon=5)
    expect = np.log(g["close"].shift(-5) / g["close"])
    ok = f["fwd_ret"].notna() & expect.notna()
    assert np.allclose(f.loc[ok, "fwd_ret"], expect[ok])


def test_scaler_is_fitted_on_train_only(panel, feats):
    """Standardised training features must have mean 0 / sd 1; the test block must
    NOT, because its statistics were never used."""
    dates = np.array(sorted(panel["date"].unique()))
    tr = panel[panel["date"].isin(dates[:800])]
    te = panel[panel["date"].isin(dates[900:])]
    xtr, xte = fold_prep(tr, te, feats)
    assert abs(float(xtr.mean().mean())) < 1e-8
    assert abs(float(xtr.std().mean()) - 1) < 1e-2
    assert not np.allclose(xte.mean().to_numpy(), 0, atol=1e-8)


def test_walk_forward_never_trains_on_the_future(panel):
    for tr_d, te_d in walk_forward_splits(panel["date"], n_folds=4, horizon=5, embargo=5):
        assert pd.Timestamp(tr_d.max()) < pd.Timestamp(te_d.min())


def test_purge_gap_is_respected(panel):
    for tr_d, te_d in walk_forward_splits(panel["date"], n_folds=4, horizon=5, embargo=5):
        gap = (pd.Timestamp(te_d.min()) - pd.Timestamp(tr_d.max())).days
        assert gap >= 5, f"purge gap only {gap} days"


def test_no_ticker_spans_a_split(panel):
    """Splits are at the date level, so a cross-section is never cut in half."""
    for tr_d, te_d in walk_forward_splits(panel["date"], n_folds=3, horizon=5, embargo=5):
        assert len(set(pd.to_datetime(tr_d)) & set(pd.to_datetime(te_d))) == 0


def test_cross_sectional_zscore_uses_one_date_only(panel):
    """Per-date standardisation must be computable from that date's rows alone."""
    d = panel["date"].unique()[600]
    day = panel[panel["date"] == d]
    if len(day) < 5:
        pytest.skip("thin cross-section")
    manual = (day["mom21"] - day["mom21"].mean()) / (day["mom21"].std() + 1e-9)
    assert np.allclose(manual.clip(-5, 5), day["mom21_cs"], atol=1e-6)


def test_assert_no_overlap_raises_on_a_bad_split(panel):
    dates = np.array(sorted(panel["date"].unique()))
    with pytest.raises(AssertionError):
        assert_no_overlap(dates[:600], dates[590:700], horizon=5, embargo=5)
