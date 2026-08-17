"""Central configuration. Everything tunable lives here or in conf/config.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "JPM", "GS", "BAC", "XOM", "CVX", "JNJ", "PFE",
    "PG", "KO", "HD", "CAT", "UNH", "T", "SPY", "QQQ", "IWM", "TLT",
    "HYG", "GLD", "XLF", "XLE",
]

# crude static sector map, used as a categorical feature (CatBoost handles it natively)
SECTOR = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "JPM": "Financials",
    "GS": "Financials", "BAC": "Financials", "XOM": "Energy", "CVX": "Energy",
    "JNJ": "Health", "PFE": "Health", "UNH": "Health", "PG": "Staples",
    "KO": "Staples", "HD": "Discretionary", "CAT": "Industrials", "T": "Comms",
    "SPY": "Index", "QQQ": "Index", "IWM": "Index", "TLT": "Rates",
    "HYG": "Credit", "GLD": "Commodity", "XLF": "SectorETF", "XLE": "SectorETF",
}


@dataclass
class Config:
    # data
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    offline: bool = True            # True -> simulated panel, no internet needed
    # signal
    horizon: int = 5                # forecast horizon in trading days
    quantile: float = 0.2           # long/short tail width
    # validation
    n_folds: int = 5
    embargo: int = 5
    n_trials: int = 25
    seed: int = 7
    # execution and risk
    half_spread_bps: float = 5.0
    impact_bps: float = 8.0
    vol_target: float = 0.10

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()
