import pandas as pd
from .base_strategy import BaseStrategy

class CombinedStrategy(BaseStrategy):
    """
    Combines HMM and ML strategies to generate trading signals.
    Supports multiple combination methods:
    - 'vote': average signals
    - 'ml_priority': ML signal preferred unless 0
    - 'hmm_priority': HMM signal preferred unless 0
    - 'and': only when both agree
    """

    def __init__(self, hmm_strategy: BaseStrategy, ml_strategy: BaseStrategy, method: str = 'and'):
        self.hmm_strategy = hmm_strategy
        self.ml_strategy = ml_strategy
        self.method = method

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        hmm_signal = self.hmm_strategy.generate_signals(X)
        ml_signal = self.ml_strategy.generate_signals(X)

        if self.method == 'vote':
            combined = (hmm_signal + ml_signal).apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            )

        elif self.method == 'ml_priority':
            combined = ml_signal.where(ml_signal != 0, hmm_signal)

        elif self.method == 'hmm_priority':
            combined = hmm_signal.where(hmm_signal != 0, ml_signal)

        elif self.method == 'and':
            combined = ((hmm_signal == ml_signal) & (hmm_signal != 0)).astype(int) * hmm_signal

        else:
            raise ValueError(f"Unsupported combination method: {self.method}")

        return pd.Series(combined.values, index=X.index)
