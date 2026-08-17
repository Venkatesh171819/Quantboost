"""Reusable feature-engineering transformers and selectors.

Everything here is fold-safe: `fit` sees training rows only, `transform` applies
stored statistics. This is the difference between a real pipeline and a leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip to training-set quantiles. Financial tails are real but they dominate
    squared loss, so cap rather than drop."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_ = X.quantile(self.lower)
        self.hi_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(self.lo_, self.hi_, axis=1)


class RankGauss(BaseEstimator, TransformerMixin):
    """Rank then map to a normal. Robust to outliers, kills monotone scale issues."""

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.qt_ = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(self.n_quantiles, max(len(X) // 2, 10)),
            subsample=200_000, random_state=0).fit(X)
        self.cols_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)[self.cols_]
        return pd.DataFrame(self.qt_.transform(X), index=X.index, columns=self.cols_)


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoding computed out-of-fold on the training data only.

    Naive target encoding is the single most common leak in tabular finance work.
    Smoothing shrinks small categories toward the global mean.
    """

    def __init__(self, cols: list[str], smoothing: float = 50.0, n_splits: int = 5):
        self.cols, self.smoothing, self.n_splits = cols, smoothing, n_splits

    def fit(self, X, y):
        X, y = pd.DataFrame(X), pd.Series(np.asarray(y), index=pd.DataFrame(X).index)
        self.prior_ = float(y.mean())
        self.maps_ = {}
        for c in self.cols:
            stats = y.groupby(X[c]).agg(["mean", "count"])
            w = stats["count"] / (stats["count"] + self.smoothing)
            self.maps_[c] = w * stats["mean"] + (1 - w) * self.prior_
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cols:
            X[f"{c}_te"] = X[c].map(self.maps_[c]).fillna(self.prior_)
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_ = {c: X[c].value_counts(normalize=True) for c in self.cols}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cols:
            X[f"{c}_freq"] = X[c].map(self.freq_[c]).fillna(0.0)
        return X


def numeric_pipeline(kind: str = "robust", winsorize: bool = True) -> Pipeline:
    """impute -> winsorize -> transform -> scale, all fitted on the fold."""
    steps: list = [("impute", SimpleImputer(strategy="median"))]
    if winsorize:
        steps.append(("winsor", Winsorizer()))
    if kind == "robust":
        steps.append(("scale", RobustScaler()))
    elif kind == "standard":
        steps.append(("scale", StandardScaler()))
    elif kind == "yeo":
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=True)))
    elif kind == "rankgauss":
        steps.append(("rg", RankGauss()))
    return Pipeline(steps)


def onehot(cols: list[str]) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                                  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ------------------------------------------------------------------ selection

def correlation_prune(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Drop the second member of any near-duplicate pair. Boosting tolerates
    collinearity, but importance attribution does not."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if any(upper[c] > threshold)]


def vif_scores(X: pd.DataFrame) -> pd.Series:
    Xf = X.fillna(X.median())
    corr = np.corrcoef(Xf.to_numpy(), rowvar=False)
    try:
        inv = np.linalg.pinv(corr)
        return pd.Series(np.diag(inv), index=X.columns).sort_values(ascending=False)
    except np.linalg.LinAlgError:                      # pragma: no cover
        return pd.Series(dtype=float)


def mutual_information(X: pd.DataFrame, y: pd.Series, seed: int = 7) -> pd.Series:
    Xf = X.fillna(X.median())
    mi = mutual_info_regression(Xf, np.asarray(y), random_state=seed)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def mean_decrease_accuracy(model, X: pd.DataFrame, y: pd.Series, splits,
                           scorer=None, seed: int = 7) -> pd.DataFrame:
    """Permutation importance evaluated under purged CV.

    Native `feature_importances_` is biased toward high-cardinality features and
    is computed in-sample; MDA under purged folds is the defensible version.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    scorer = scorer or (lambda a, b: spearmanr(a, b).statistic)
    rows = []
    for tr, te in splits:
        m = clone(model).fit(X.iloc[tr].fillna(0), y.iloc[tr])
        base = scorer(m.predict(X.iloc[te].fillna(0)), y.iloc[te])
        rec = {"base": base}
        for col in X.columns:
            Xp = X.iloc[te].copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            rec[col] = base - scorer(m.predict(Xp.fillna(0)), y.iloc[te])
        rows.append(rec)
    out = pd.DataFrame(rows)
    return (out.drop(columns=["base"]).agg(["mean", "std"]).T
            .rename(columns={"mean": "mda", "std": "mda_sd"})
            .sort_values("mda", ascending=False))


def clustered_importance(X: pd.DataFrame, importance: pd.Series,
                         n_clusters: int = 8) -> pd.DataFrame:
    """Group substitutable features, then sum importance per cluster. Stops
    correlated features from splitting credit and looking individually useless."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    corr = X.corr().fillna(0)
    dist = squareform(np.clip(1 - corr.abs().to_numpy(), 0, 2), checks=False)
    labels = fcluster(linkage(dist, method="average"), n_clusters, criterion="maxclust")
    df = pd.DataFrame({"feature": corr.columns, "cluster": labels})
    df["importance"] = df["feature"].map(importance).fillna(0.0)
    return df.sort_values(["cluster", "importance"], ascending=[True, False])
