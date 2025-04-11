from abc import ABC, abstractmethod
import pandas as pd

class BaseMetrics(ABC):
    def __init__(self, equity_curve: pd.Series, trades: pd.DataFrame = None, trading_fee: float = 0.0006, risk_free_rate: float = 0.0):
        """
        :param equity_curve: pandas Series with portfolio value over time (indexed by date/time).
        :param trades: pandas DataFrame or list of trades with at least 'pnl', 'entry_time', 'exit_time' columns.
        :param trading_fee: trading fee as a decimal (e.g., 0.0006 for 0.06%).
        :param risk_free_rate: annual risk-free rate used in Sharpe calculation (e.g., 0.02 = 2%).
        """
        self.equity = equity_curve
        self.returns = self.equity.pct_change().dropna()
        self.trades = pd.DataFrame(trades) if trades is not None else None
        self.trading_fee = trading_fee
        self.risk_free_rate = risk_free_rate

        if self.trades is not None:
            required_columns = ['pnl', 'entry_time', 'exit_time']
            missing_columns = [col for col in required_columns if col not in self.trades.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in trades: {', '.join(missing_columns)}")


    @abstractmethod
    def all_metrics(self) -> dict:
        """Return a dictionary of all metrics."""
        pass
