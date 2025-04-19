import pandas as pd
from typing import Any
import numpy as np
from .base_strategy import BaseStrategy
from ..config import BaseConfig
import torch

class MLStrategy(BaseStrategy):
    """
    ML-based strategy that uses a trained model to generate trading signals.
    """

    def __init__(self, model: Any):
        """
        :param model: A trained ML model
        """
        self.model = model

    def generate_signals(self, X: pd.DataFrame, threshold: float = None) -> pd.Series:
        """
        Generate buy/hold/sell signals using the model's predicted probabilities.
        
        :param X: DataFrame of features or price data with DateTimeIndex
        :param threshold: Optional threshold for buy/sell signals
        :return: Trading signals aligned with input index
        """
        if threshold is None:
            return self.model.predict(X)
        else:
            return self.model.predict(X, threshold=threshold)