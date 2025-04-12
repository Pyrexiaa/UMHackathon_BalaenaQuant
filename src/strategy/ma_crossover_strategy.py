import pandas as pd
from .base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    Generates buy/sell signals when short-term MA crosses above/below long-term MA.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        :param short_window: Window size for the short-term moving average.
        :param long_window: Window size for the long-term moving average.
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on moving average crossover.

        :param X: DataFrame containing at least a 'Close' column
        :return: pd.Series of signals (1 = Buy, -1 = Sell, 0 = Hold)
        """
        # Compute moving averages
        short_ma = X['Close'].rolling(window=self.short_window, min_periods=1).mean()
        long_ma = X['Close'].rolling(window=self.long_window, min_periods=1).mean()

        # Initialize signal series
        signal = pd.Series(0, index=X.index)

        # Buy signal: short crosses above long
        crossover_buy = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        signal[crossover_buy] = 1

        # Sell signal: short crosses below long
        crossover_sell = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        signal[crossover_sell] = -1

        return signal
