"""Model zoo: XGBoost, CatBoost, AdaBoost, plus a stacked meta-learner.

AdaBoost is deliberately included. Its exponential/linear loss is sensitive to the
fat tails of financial returns, so it usually loses to the modern boosters. That
comparison is the point: an ablation you can explain beats a leaderboard number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

DEFAULT_XGB = dict(n_estimators=450, max_depth=4, learning_rate=0.035, subsample=0.8,
                   colsample_bytree=0.6, min_child_weight=12.0, reg_lambda=3.0,
                   reg_alpha=0.05)

DEFAULT_CAT = dict(iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=6.0,
                   random_strength=1.0, bagging_temperature=0.5)

DEFAULT_ADA = dict(n_estimators=250, learning_rate=0.05, base_depth=3)


def make_regressors(xgb_params: dict | None = None, cat_params: dict | None = None,
                    ada_params: dict | None = None, seed: int = 7,
                    monotone: dict | None = None) -> dict:
    xgb_params = {**DEFAULT_XGB, **(xgb_params or {})}
    cat_params = {**DEFAULT_CAT, **(cat_params or {})}
    ada_params = {**DEFAULT_ADA, **(ada_params or {})}
    out: dict = {}

    if HAS_XGB:
        kw = dict(objective="reg:pseudohubererror", tree_method="hist", n_jobs=-1,
                  random_state=seed, **xgb_params)
        if monotone:
            kw["monotone_constraints"] = monotone
        out["XGBoost"] = XGBRegressor(**kw)

    if HAS_CAT:
        out["CatBoost"] = CatBoostRegressor(loss_function="RMSE", verbose=0,
                                            random_seed=seed,
                                            allow_writing_files=False, **cat_params)

    depth = ada_params.pop("base_depth", 3)
    out["AdaBoost"] = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=depth, random_state=seed),
        loss="linear", random_state=seed, **ada_params)

    if not HAS_XGB and not HAS_CAT:
        out["Ridge (fallback)"] = Ridge(alpha=5.0)
    return out


def make_classifiers(seed: int = 7) -> dict:
    """For triple-barrier direction and for the meta-labelling stage."""
    out: dict = {}
    if HAS_XGB:
        out["XGBoost"] = XGBClassifier(objective="binary:logistic", tree_method="hist",
                                       n_estimators=400, max_depth=4, learning_rate=0.04,
                                       subsample=0.8, colsample_bytree=0.6,
                                       min_child_weight=10, reg_lambda=3.0,
                                       n_jobs=-1, random_state=seed, eval_metric="logloss")
    if HAS_CAT:
        out["CatBoost"] = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.04,
                                             l2_leaf_reg=6.0, verbose=0, random_seed=seed,
                                             allow_writing_files=False)
    out["AdaBoost"] = AdaBoostClassifier(n_estimators=300, learning_rate=0.05,
                                         random_state=seed)
    if not HAS_XGB and not HAS_CAT:
        out["Logistic (fallback)"] = LogisticRegression(max_iter=2000)
    return out


def fit_with_weights(model, X, y, w=None):
    try:
        return model.fit(X, y, sample_weight=w) if w is not None else model.fit(X, y)
    except TypeError:                                   # estimator ignores weights
        return model.fit(X, y)


def rank_average(preds: pd.DataFrame) -> pd.Series:
    """Rank-average ensemble: scale-free, so a badly calibrated member cannot
    dominate the blend."""
    return preds.rank(pct=True).mean(axis=1) - 0.5


def stack(oof_preds: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> tuple[Ridge, pd.Series]:
    """Ridge meta-learner on purged out-of-fold predictions only."""
    m = Ridge(alpha=alpha, fit_intercept=True).fit(oof_preds.fillna(0), y)
    return m, pd.Series(m.coef_, index=oof_preds.columns)
