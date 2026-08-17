"""Hyperparameter search and overfitting statistics.

Two non-negotiables:
  1. the objective is economic (rank IC, stability-penalised), not RMSE;
  2. every reported Sharpe is accompanied by a Deflated Sharpe Ratio that knows
     how many configurations you tried.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .models import DEFAULT_XGB, HAS_XGB

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


def rank_ic(df: pd.DataFrame, pred_col: str = "pred", target_col: str = "target") -> pd.Series:
    """Spearman correlation between forecast and outcome, per date."""
    def _ic(g):
        if len(g) < 5:
            return np.nan
        return stats.spearmanr(g[pred_col], g[target_col]).statistic
    return df.groupby("date", group_keys=False).apply(_ic).dropna()


def fold_prep(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]):
    """Median impute + standardise, fitted on TRAIN ONLY. The whole leak surface
    of a tabular pipeline lives in this function, so it is deliberately tiny."""
    med = train[feats].median()
    xtr, xte = train[feats].fillna(med), test[feats].fillna(med)
    mu, sd = xtr.mean(), xtr.std().replace(0, 1)
    return (xtr - mu) / sd, (xte - mu) / sd


def suggest_xgb(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 900, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 7),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 60, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 40, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
    )


def suggest_cat(trial) -> dict:
    return dict(
        iterations=trial.suggest_int("iterations", 250, 900, step=50),
        depth=trial.suggest_int("depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.2, 3.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.5),
    )


def tune_xgb(panel: pd.DataFrame, feats: list[str], n_trials: int = 25, horizon: int = 5,
             embargo: int = 5, seed: int = 7, inner_frac: float = 0.7,
             turnover_penalty: float = 0.0):
    """Nested search: tuning happens inside the first training window only, so the
    walk-forward test blocks stay untouched by the selection process.

    Objective = mean out-of-fold rank IC / its dispersion. Level alone rewards a
    single lucky fold; the ratio rewards something you could actually run.
    """
    if not (HAS_OPTUNA and HAS_XGB) or n_trials <= 0:
        return dict(DEFAULT_XGB), None

    from xgboost import XGBRegressor
    dates = np.array(sorted(panel["date"].unique()))
    cut = int(len(dates) * inner_frac)
    tr_d, va_d = dates[:max(cut - (horizon + embargo), 50)], dates[cut:]
    tr, va = panel[panel["date"].isin(tr_d)], panel[panel["date"].isin(va_d)]
    if len(tr) < 500 or len(va) < 200:
        return dict(DEFAULT_XGB), None
    xtr, xva = fold_prep(tr, va, feats)

    def objective(trial):
        p = suggest_xgb(trial)
        m = XGBRegressor(objective="reg:pseudohubererror", tree_method="hist",
                         n_jobs=-1, random_state=seed, **p)
        m.fit(xtr, tr["target"])
        d = pd.DataFrame({"date": va["date"].to_numpy(), "pred": m.predict(xva),
                          "target": va["target"].to_numpy()})
        ic = rank_ic(d)
        if len(ic) < 20:
            return -1.0
        score = float(ic.mean() / (ic.std() + 1e-6))
        if turnover_penalty:
            wide = d.assign(t=d["pred"]).pivot_table(index="date", values="pred",
                                                     aggfunc="mean")
            score -= turnover_penalty * float(wide.diff().abs().mean().iloc[0])
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return {**DEFAULT_XGB, **study.best_params}, study


# ------------------------------------------------- overfitting statistics

def deflated_sharpe(sr: float, n_obs: int, n_trials: int, sr_variance: float,
                    skew: float, kurtosis: float) -> float:
    """Bailey & Lopez de Prado (2014). Probability the observed Sharpe would survive
    the number of trials, the sample length and the non-normality of returns.

    Interpretation: below ~0.95 you have not demonstrated skill, you have searched.
    """
    gamma, e = 0.5772156649015329, math.e
    t = max(int(n_trials), 2)
    z1, z2 = stats.norm.ppf(1 - 1 / t), stats.norm.ppf(1 - 1 / (t * e))
    sr_star = math.sqrt(max(sr_variance, 1e-12)) * ((1 - gamma) * z1 + gamma * z2)
    den = math.sqrt(max(1 - skew * sr + (kurtosis - 1) / 4 * sr ** 2, 1e-9))
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(max(n_obs - 1, 1)) / den))


def probability_of_backtest_overfitting(perf_matrix: pd.DataFrame) -> float:
    """PBO via combinatorially symmetric CV: how often the in-sample best
    configuration lands below median out of sample.

    `perf_matrix`: rows = time blocks, columns = candidate configurations.
    """
    import itertools
    M = perf_matrix.dropna(axis=1, how="any")
    blocks = list(M.index)
    if M.shape[1] < 2 or len(blocks) < 4:
        return float("nan")
    half = len(blocks) // 2
    losses = 0
    trials = 0
    for combo in itertools.combinations(blocks, half):
        is_ = M.loc[list(combo)].mean()
        oos = M.drop(index=list(combo)).mean()
        best = is_.idxmax()
        rank = oos.rank(pct=True)[best]
        losses += int(rank < 0.5)
        trials += 1
        if trials >= 200:
            break
    return losses / max(trials, 1)


def reality_check(strategy: pd.Series, benchmarks: pd.DataFrame, n_boot: int = 500,
                  seed: int = 7) -> float:
    """White's Reality Check p-value: is the best strategy better than the best
    benchmark once you account for having looked at many of them?"""
    rng = np.random.default_rng(seed)
    d = benchmarks.apply(lambda b: strategy - b)
    stat = d.mean().max() * math.sqrt(len(strategy))
    boots = []
    n = len(d)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = d.iloc[idx]
        boots.append(((s.mean() - d.mean()).max()) * math.sqrt(n))
    return float(np.mean(np.array(boots) >= stat))
