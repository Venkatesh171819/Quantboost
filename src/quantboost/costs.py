"""Transaction costs. A backtest without these is a plot, not a result."""
from __future__ import annotations

import numpy as np
import pandas as pd


def spread_cost(turnover: pd.Series, half_spread_bps: float = 5.0) -> pd.Series:
    """Linear component: you cross half the spread on every unit traded."""
    return turnover.abs() * half_spread_bps / 1e4


def impact_cost(turnover: pd.Series, impact_bps: float = 8.0, exponent: float = 0.5) -> pd.Series:
    """Almgren-Chriss style temporary impact: cost grows with the square root of
    participation, so size is punished superlinearly in total dollars."""
    return np.power(turnover.abs().clip(lower=0), exponent) * impact_bps / 1e4


def borrow_cost(short_exposure: pd.Series, annual_bps: float = 50.0,
                days: int = 252) -> pd.Series:
    return short_exposure.abs() * (annual_bps / 1e4) / days


def total_cost(turnover: pd.Series, short_exposure: pd.Series | None = None,
               half_spread_bps: float = 5.0, impact_bps: float = 8.0,
               borrow_bps: float = 50.0) -> pd.Series:
    c = spread_cost(turnover, half_spread_bps) + impact_cost(turnover, impact_bps)
    if short_exposure is not None:
        c = c + borrow_cost(short_exposure, borrow_bps)
    return c


def break_even_ic(turnover_annual: float, cost_per_unit_bps: float,
                  vol_annual: float = 0.10) -> float:
    """Roughly the IC you need before costs eat the whole signal. Sanity anchor:
    if your break-even IC is above your measured IC, stop building."""
    return float(turnover_annual * cost_per_unit_bps / 1e4 / max(vol_annual, 1e-9))
