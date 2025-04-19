# How to Add a Custom Strategy to QuantPilot

QuantPilot supports custom trading strategies by extending the `BaseStrategy` class. This gives you flexibility to design rule-based or ML-based strategies and run them through the backtesting engine.

---

## Step 1: Create a Custom Strategy Class

Create a new strategy class that inherits from `BaseStrategy`. You must implement the `generate_signals` method.

```python
# my_strategy.py
import pandas as pd
from quantpilot.strategy.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        # Example: Simple Moving Average (SMA) crossover strategy
        short_ma = X["close"].rolling(window=20).mean()
        long_ma = X["close"].rolling(window=50).mean()

        signals = pd.Series(index=X.index, dtype=int)
        signals[short_ma > long_ma] = 2  # Buy
        signals[short_ma < long_ma] = 0  # Sell
        signals[short_ma == long_ma] = 1  # Hold

        return signals.fillna(1)  # Default to Hold
```

## Step 2: Use Custom Strategy in Backtester
You can now plug your custom strategy into the backtesting pipeline:
```python
from quantpilot.backtester import Backtester
from my_strategy import MyCustomStrategy
from your_data_loader import load_data  # Replace with actual data loader

df = load_data()
strategy = MyCustomStrategy()

bt = Backtester(data=df, strategy=strategy, strategy_name="MyCustomStrategy")
bt.run()
bt.plot_results()
```

## Tips for Strategy Development
Ensure signals are time-aligned with the data index.
Use `.fillna(1)` to default missing signals to "Hold".
Add logic to incorporate features, thresholds, or rolling statistics.
Combine with `add_hmm_features`, `add_nlp_sentiment_score`, or on-chain indicators for advanced strategies.