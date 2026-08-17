"""Cross-validation for overlapping financial labels.

Plain KFold on a time series is an instant reject in a quant interview: it trains
on the future and, because label windows overlap, it leaks even when it does not
obviously look ahead. Everything here purges and embargoes.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """Contiguous test blocks; training rows whose label window overlaps the test
    block are purged, plus a forward embargo to kill serial correlation spillover.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 pct_embargo: float = 0.01):
        self.n_splits, self.t1, self.pct_embargo = n_splits, t1, pct_embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        idx = np.arange(n)
        emb = int(n * self.pct_embargo)
        t1 = self.t1 if self.t1 is not None else pd.Series(np.arange(n), index=np.arange(n))
        t1 = pd.Series(np.asarray(t1), index=np.arange(n))
        for block in np.array_split(idx, self.n_splits):
            start, end = block[0], block[-1] + 1
            test = idx[start:end]
            test_start = t1.iloc[start]
            # purge: drop training rows whose label window reaches into the test block
            left = idx[:start][t1.iloc[:start].to_numpy() < test_start]
            # embargo: skip a buffer after the test block before training resumes
            right = idx[min(end + emb, n):]
            yield np.concatenate([left, right]).astype(int), test


def walk_forward_splits(dates, n_folds: int = 5, horizon: int = 5, embargo: int = 5,
                        min_train_days: int = 500, expanding: bool = True):
    """Date-level expanding (or rolling) walk-forward with a purge + embargo gap.

    Returns a list of (train_dates, test_dates). Work at the date level, not the
    row level, so a whole cross-section stays on one side of the split.
    """
    uniq = np.array(sorted(pd.unique(pd.to_datetime(dates))))
    if len(uniq) < min_train_days + n_folds * 40:
        min_train_days = max(250, int(len(uniq) * 0.4))
    span = max((len(uniq) - min_train_days) // n_folds, 1)
    gap = horizon + embargo
    splits = []
    for k in range(n_folds):
        t0 = min_train_days + k * span
        t1 = len(uniq) if k == n_folds - 1 else min(t0 + span, len(uniq))
        if t1 - t0 < 20:
            continue
        train_end = max(t0 - gap, 10)
        train_start = 0 if expanding else max(train_end - min_train_days, 0)
        splits.append((uniq[train_start:train_end], uniq[t0:t1]))
    return splits


def combinatorial_purged_splits(dates, n_groups: int = 6, n_test: int = 2,
                               horizon: int = 5, embargo: int = 5):
    """CPCV: every combination of `n_test` blocks as the test set, giving many
    backtest paths instead of one. You report the distribution of Sharpe, not a point.
    """
    uniq = np.array(sorted(pd.unique(pd.to_datetime(dates))))
    groups = np.array_split(uniq, n_groups)
    gap = pd.Timedelta(days=int((horizon + embargo) * 1.5))
    out = []
    for combo in itertools.combinations(range(n_groups), n_test):
        test = np.concatenate([groups[i] for i in combo])
        lo, hi = test.min() - gap, test.max() + gap
        train = np.concatenate([g for i, g in enumerate(groups) if i not in combo])
        train = train[(train < lo) | (train > hi)]
        if len(train) > 250 and len(test) > 40:
            out.append((train, test))
    return out


def assert_no_overlap(train_dates, test_dates, horizon: int, embargo: int) -> None:
    """Guard used by the test suite: no training date may sit inside the purged gap."""
    tr, te = pd.to_datetime(pd.Index(train_dates)), pd.to_datetime(pd.Index(test_dates))
    gap = pd.Timedelta(days=int((horizon + embargo) * 1.5))
    bad = tr[(tr >= te.min() - gap) & (tr <= te.max() + gap)]
    if len(bad):
        raise AssertionError(f"{len(bad)} training dates fall inside the purge window")
