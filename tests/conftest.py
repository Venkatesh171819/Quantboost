import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from quantboost.data import simulate_panel
from quantboost.features import build_panel, feature_columns

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


@pytest.fixture(scope="session")
def prices():
    return simulate_panel(TICKERS, "2016-01-01", "2021-12-31", seed=11)


@pytest.fixture(scope="session")
def panel(prices):
    return build_panel(prices, horizon=5)


@pytest.fixture(scope="session")
def feats(panel):
    return [c for c in feature_columns() if c in panel.columns]
