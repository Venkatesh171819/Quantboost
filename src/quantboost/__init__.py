"""QuantBoost: regime-aware boosted return forecasting with a risk-managed overlay."""
__version__ = "1.0.0"

from . import (backtest, costs, cv, data, features, fe_toolkit, labeling, models,
               plotting, portfolio, risk, tsa, tuning)

__all__ = ["backtest", "costs", "cv", "data", "features", "fe_toolkit", "labeling",
           "models", "plotting", "portfolio", "risk", "tsa", "tuning", "config"]
