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

    def __init__(self, model: Any, threshold: float = BaseConfig.THRESHOLD):
        """
        :param model: A trained ML model
        :param threshold: Confidence threshold for signals
        """
        self.model = model
        self.threshold = threshold

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate buy/hold/sell signals using the model's predicted probabilities.
        
        :param X: DataFrame of features or price data with DateTimeIndex
        :return: Trading signals aligned with input index
        """
        return self.model.predict(X)