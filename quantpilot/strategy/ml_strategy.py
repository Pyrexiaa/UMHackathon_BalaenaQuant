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

    def __init__(self, model: Any, buy_threshold: float = BaseConfig.BUY_THRESHOLD, sell_threshold: float = BaseConfig.SELL_THRESHOLD):
        """
        :param model: A trained ML model
        :param threshold: Confidence threshold for signals
        """
        self.model = model
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate buy/hold/sell signals using the model's predicted probabilities.
        
        :param X: DataFrame of features or price data with DateTimeIndex
        :return: Trading signals (2=Buy, 1=Hold, 0=Sell) aligned with input index
        """
        # Get model predictions
        try:
            probs = self.model.predict(X)
            probs = np.asarray(probs)
            
            # Validate predictions
            if len(probs) == 0:
                return pd.Series(BaseConfig.HOLD_SIGNAL, index=X.index)
                
            if probs.shape[1] != 3:  # Should have 3 classes (Buy/Hold/Sell)
                raise ValueError(f"Expected 3 output probabilities, got {probs.shape[1]}")
                
            if not np.allclose(probs.sum(axis=1), 1, rtol=1e-3):
                raise ValueError("Probabilities must sum to 1")
                
        except Exception as e:
            raise ValueError(f"Model prediction failed: {str(e)}")
        
        # Generate signals
        signals = []
        valid_indices = []  # To track which indices have valid predictions
        
        # The model's predictions will be shorter than input due to windowing
        # So we need to align them properly
        window_size = getattr(self.model, 'window_size', BaseConfig.WINDOW_SIZE)
        start_idx = window_size  # First prediction corresponds to this index
        
        # Initialize with HOLD signals for the warmup period
        for i in range(start_idx):
            signals.append(BaseConfig.HOLD_SIGNAL)
            valid_indices.append(X.index[i])
        
        # Process model predictions
        for i, p in enumerate(probs, start=start_idx):
            if i >= len(X):
                break  # Handle case where we have extra predictions
                
            if p[BaseConfig.BUY_SIGNAL] > self.buy_threshold:
                signals.append(BaseConfig.BUY_SIGNAL)
            elif p[BaseConfig.SELL_SIGNAL] > self.sell_threshold:
                signals.append(BaseConfig.SELL_SIGNAL)
            else:
                signals.append(BaseConfig.HOLD_SIGNAL)
            valid_indices.append(X.index[i])
        
        # Handle case where we didn't get enough predictions
        while len(signals) < len(X):
            signals.append(BaseConfig.HOLD_SIGNAL)
            valid_indices.append(X.index[len(signals)-1])
        
        return pd.Series(signals, index=X.index)