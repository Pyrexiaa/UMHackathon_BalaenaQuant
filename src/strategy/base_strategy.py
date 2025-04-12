from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """

    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals from input data.
        
        :params X: DataFrame of features or price data.
        :return: Series of trading signals (1 = Buy, -1 = Sell, 0 = Hold)
        """
        pass