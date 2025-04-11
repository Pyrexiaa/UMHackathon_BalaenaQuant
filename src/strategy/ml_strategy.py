import pandas as pd
from .base_strategy import BaseStrategy
from typing import Any

class MLStrategy(BaseStrategy):
    """
    ML-based strategy that uses a trained model to generate trading signals.
    """

    def __init__(self, model: Any, threshold: float = 0.5):
        """
        :params model: A trained ML model with a predict_proba() method.
        :params threshold: Threshold to convert probability to buy/sell signal.
        """
        self.model = model
        self.threshold = threshold

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate buy/hold signals using the model's predicted probabilities.
        
        :params X: DataFrame of features or price data.
        :returns: pd.Series: Signals: 1 (Buy), 0 (Hold), -1(Sell)
        """
        probs = self.model.predict_proba(X)[:, 1]  # Probability of class '1' (buy)
        signals = (probs > self.threshold).astype(int)
        return pd.Series(signals, index=X.index).shift(1).fillna(0)
