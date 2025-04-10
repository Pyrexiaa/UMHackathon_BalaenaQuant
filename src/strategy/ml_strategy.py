import numpy as np
import pandas as pd
from src.strategy.base_strategy import Strategy

class MLThresholdStrategy(Strategy):
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def generate_signals(self, probs: np.ndarray, prices: pd.Series, **kwargs) -> pd.DataFrame:
        min_length = min(len(probs), len(prices))
        probs = probs[:min_length]
        prices = prices.iloc[:min_length]

        signals = pd.DataFrame(index=pd.RangeIndex(start=0, stop=min_length, step=1))
        signals['price'] = prices.values
        signals['signal'] = 0

        buy_indices = np.where(probs[:, 0] > self.threshold)[0]
        sell_indices = np.where(probs[:, 1] > self.threshold)[0]

        signals.loc[buy_indices, 'signal'] = 1
        signals.loc[sell_indices, 'signal'] = -1
        signals['trade_type'] = signals['signal'].map({1: "BUY", -1: "SELL", 0: "NONE"})

        signals['entry_price'] = signals['price']
        signals['stop_loss'] = signals.apply(
            lambda x: x['entry_price'] * (1 - 0.005) if x['signal'] == 1 else x['entry_price'] * (1 + 0.005),
            axis=1
        )
        signals['take_profit'] = signals.apply(
            lambda x: x['entry_price'] * (1 + 0.015) if x['signal'] == 1 else x['entry_price'] * (1 - 0.015),
            axis=1
        )

        filtered_signals = signals[signals['signal'] != 0]
        if filtered_signals.empty:
            print("No valid signals above the threshold.")
            return pd.DataFrame()

        filtered_signals.index = prices.index[:len(filtered_signals)]
        return filtered_signals[['trade_type', 'entry_price', 'stop_loss', 'take_profit']]
