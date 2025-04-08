import pandas as pd
from strategy_base import Strategy

class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, short_window=10, long_window=30):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, probs: None, prices: pd.Series, **kwargs) -> pd.DataFrame:
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices
        signals['short_ma'] = prices.rolling(window=self.short_window).mean()
        signals['long_ma'] = prices.rolling(window=self.long_window).mean()

        signals['signal'] = 0
        signals['signal'][self.short_window:] = (
            (signals['short_ma'][self.short_window:] > signals['long_ma'][self.short_window:]).astype(int) * 2 - 1
        )

        signals['trade_type'] = signals['signal'].map({1: "BUY", -1: "SELL", 0: "NONE"})
        signals['entry_price'] = signals['price']
        signals['stop_loss'] = signals['entry_price'] * 0.99
        signals['take_profit'] = signals['entry_price'] * 1.03

        filtered_signals = signals[signals['signal'] != 0]
        return filtered_signals[['trade_type', 'entry_price', 'stop_loss', 'take_profit']]
