import pandas as pd
from .base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    Generates buy/sell signals when short-term MA crosses above/below long-term MA.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        :params short_window: Window size for the short-term moving average.
        :params long_window: Window size for the long-term moving average.
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on moving average crossover.

        Args:
        :params: DataFrame of price data
        :returns: pd.Series: Series of signals (1 = Buy, -1 = Sell, 0 = Hold)
        """
        short_ma = X['Close'].rolling(window=self.short_window).mean()
        long_ma = X['Close'].rolling(window=self.long_window).mean()

        signal = pd.Series(0, index=X.index)
        signal[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))] = 1
        signal[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))] = -1

        return signal
