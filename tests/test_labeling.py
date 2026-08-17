import numpy as np
import pandas as pd

from quantboost.labeling import (average_uniqueness, meta_labels, sample_weights,
                                 triple_barrier)


def test_profit_take_is_detected():
    close = pd.Series(np.r_[100.0, np.linspace(100, 130, 20)])
    vol = pd.Series(0.01, index=close.index)
    out = triple_barrier(close, vol, pt=2.0, sl=1.0, max_hold=10)
    assert out["label"].iloc[0] == 1.0


def test_stop_loss_is_detected():
    close = pd.Series(np.r_[100.0, np.linspace(100, 70, 20)])
    vol = pd.Series(0.01, index=close.index)
    out = triple_barrier(close, vol, pt=2.0, sl=1.0, max_hold=10)
    assert out["label"].iloc[0] == -1.0


def test_time_barrier_gives_zero():
    close = pd.Series(100 + np.zeros(30))
    vol = pd.Series(0.02, index=close.index)
    out = triple_barrier(close, vol, max_hold=10)
    assert out["label"].iloc[0] == 0.0


def test_barriers_scale_with_volatility():
    close = pd.Series(100 * np.exp(np.cumsum(np.r_[0, np.full(20, 0.004)])))
    tight = triple_barrier(close, pd.Series(0.005, index=close.index), max_hold=15)
    loose = triple_barrier(close, pd.Series(0.05, index=close.index), max_hold=15)
    assert abs(tight["label"].iloc[0]) >= abs(loose["label"].iloc[0])


def test_uniqueness_is_between_zero_and_one():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    t1 = pd.Series(idx, index=idx).shift(-5).ffill()
    u = average_uniqueness(t1)
    assert ((u > 0) & (u <= 1.0001)).all()


def test_sample_weights_are_normalised():
    w = sample_weights(None, 200)
    assert abs(w.mean() - 1) < 1e-9 and (w > 0).all()


def test_meta_labels_flag_correct_bets():
    side = pd.Series([1, 1, -1, -1])
    real = pd.Series([0.02, -0.02, -0.02, 0.02])
    assert meta_labels(side, real).tolist() == [1, 0, 1, 0]
