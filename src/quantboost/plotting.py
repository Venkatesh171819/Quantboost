"""Matplotlib charts for the notebooks, styled to match the Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd

PAPER, INK, INK2, RULE = "#F8F3EC", "#2B2320", "#6B5F58", "#DDD3C8"
ACCENT, UP, DOWN = "#8C2F26", "#1F6F63", "#B4453B"


def use_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "grid.color": RULE,
        "axes.grid": True, "grid.alpha": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "figure.dpi": 110,
        "axes.prop_cycle": mpl.cycler(color=[ACCENT, INK, UP, INK2, DOWN]),
    })


def plot_equity(curves: dict, title: str = "Cumulative net growth of 1.00", ax=None):
    import matplotlib.pyplot as plt
    use_style()
    ax = ax or plt.subplots(figsize=(11, 4.2))[1]
    for i, (name, eq) in enumerate(curves.items()):
        ax.plot(eq.index, eq.to_numpy(), label=name, lw=2.0 if i == 0 else 1.2)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, ncols=len(curves))
    return ax


def plot_drawdown(eq: pd.Series, ax=None):
    import matplotlib.pyplot as plt
    use_style()
    ax = ax or plt.subplots(figsize=(11, 2.4))[1]
    dd = (eq / eq.cummax() - 1) * 100
    ax.fill_between(dd.index, dd.to_numpy(), 0, color=DOWN, alpha=0.25)
    ax.plot(dd.index, dd.to_numpy(), color=DOWN, lw=0.9)
    ax.set_title("Drawdown, %", loc="left")
    return ax


def plot_importance(imp: pd.Series, top: int = 20, ax=None):
    import matplotlib.pyplot as plt
    use_style()
    ax = ax or plt.subplots(figsize=(7, 6))[1]
    s = imp.head(top)[::-1]
    ax.barh(s.index, s.to_numpy(), color=INK2)
    ax.set_title("Feature importance", loc="left")
    return ax


def plot_ic(ic: pd.Series, window: int = 63, ax=None):
    import matplotlib.pyplot as plt
    use_style()
    ax = ax or plt.subplots(figsize=(11, 3))[1]
    ax.bar(ic.index, ic.to_numpy(), color=RULE, width=1.0)
    ax.plot(ic.index, ic.rolling(window).mean().to_numpy(), color=ACCENT, lw=1.6,
            label=f"{window}d mean")
    ax.axhline(0, color=INK, lw=0.8)
    ax.legend(frameon=False)
    ax.set_title("Daily rank information coefficient", loc="left")
    return ax


def plot_monthly_heatmap(returns: pd.Series, ax=None):
    import matplotlib.pyplot as plt
    use_style()
    idx = returns.index.to_series()
    m = returns.groupby([idx.dt.year, idx.dt.month]).sum().unstack()
    ax = ax or plt.subplots(figsize=(9, 0.42 * max(len(m), 6) + 1.5))[1]
    im = ax.imshow(m.to_numpy(), cmap="RdYlGn", vmin=-0.06, vmax=0.06, aspect="auto")
    ax.set_xticks(range(m.shape[1]), [pd.Timestamp(2000, c, 1).strftime("%b")
                                      for c in m.columns])
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, shrink=0.7, label="monthly return")
    ax.set_title("Monthly net returns", loc="left")
    return ax
