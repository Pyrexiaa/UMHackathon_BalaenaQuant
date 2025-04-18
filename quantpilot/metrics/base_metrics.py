from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class BaseMetrics(ABC):
    """
    Abstract base class for performance metrics calculation in trading systems.
    Provides core metrics that all strategy evaluations should implement.
    """
    def __init__(self,
                 equity: pd.Series,
                 drawdown: pd.Series,
                 returns: pd.Series,
                 trades: Optional[pd.DataFrame] = None,
                 records: Optional[pd.DataFrame] = None,
                 risk_free_rate: float = 0.0,
                 trading_fee: float = 0.0006,
                 mode: str = "arithmetic",
                 signals: Optional[pd.Series] = None):
        """
        Initialize base metrics calculator.
        
        :param equity: Series of portfolio values over time
        :param drawdown: Series of drawdown values over time
        :param returns: Series of daily returns
        :param trades: DataFrame containing trade records
        :param records: DataFrame containing trade records (optional)
        :param risk_free_rate: Annual risk-free rate for risk-adjusted metrics
        :param trading_fee: Fee per trade (default 0.06%)
        :param mode: Mode of calculation (arithmetic or geometric)
        :param signals: Series of trading signals (for frequency calculations)
        """
        self.equity = equity
        self.drawdown = drawdown
        self.returns = returns
        self.trades = trades
        self.records = records
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.mode = mode
        self.signals =signals
        
    @abstractmethod
    def all_metrics(self) -> dict:
        """Return a dictionary of all metrics."""
        pass

    def get_metrics(self, names: list = None) -> dict:
        """Return selected metrics."""
        all_metrics = self.all_metrics()
        if names is None:
            return all_metrics
        return {k: all_metrics[k] for k in names if k in all_metrics}
