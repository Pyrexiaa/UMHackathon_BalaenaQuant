import pandas as pd
from typing import Any
import numpy as np
from .base_strategy import BaseStrategy
from ..config import Config

class MLStrategy(BaseStrategy):
    """
    ML-based strategy that uses a trained model to generate trading signals.
    """

    def __init__(self, model: Any, buy_threshold: float = Config.BUY_THRESHOLD, sell_threshold: float = Config.SELL_THRESHOLD):
        """
        :params model: A trained ML model
        :params threshold: Confidence threshold for signals
        """
        self.model = model
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate buy/hold/sell signals using the model's predicted probabilities.
        
        :params X: DataFrame of features or price data.
        :returns: pd.Series: Signals: 2 (Buy), 1 (Hold), 0(Sell)
        """
        probs = self.model.predict(X)
            
        # Ensure signals are only created for the valid length of the input data
        signals = []
        for p in probs:
            p = np.array(p)
            if p[Config.BUY_SIGNAL] > self.buy_threshold:
                signals.append(Config.BUY_SIGNAL)
            elif p[Config.SELL_SIGNAL] > self.sell_threshold:
                signals.append(Config.SELL_SIGNAL)
            else:
                signals.append(Config.HOLD_SIGNAL)
        
        # Ensure the signals length matches the portfolio length
        while len(signals) < len(X):
            signals.append(Config.HOLD_SIGNAL)  # Padding with a default value if needed
        
        return pd.Series(signals, index=X.index)  # Ensure index alignment