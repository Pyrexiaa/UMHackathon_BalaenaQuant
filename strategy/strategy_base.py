from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, probs: np.ndarray, prices: pd.Series, **kwargs) -> pd.DataFrame:
        """Generate trading signals given model probabilities and price data."""
        pass

def run_strategy(strategy, probs, prices, **kwargs):
    return strategy.generate_signals(probs, prices, **kwargs)