# 📘 Strategy Module Documentation

This module provides a structured and extensible way to implement and test trading strategies using a backtesting framework. It includes a base abstract class and two example strategy implementations based on moving averages.

---

### 🧱 `BaseStrategy` — Abstract Base Class (`base_strategy.py`)
Provides a common interface for all trading strategies.

### Core Functionality

```python
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self) -> pd.Series:
```

Abstract Method:

`generate_signals(self)` -> pd.Series: All derived strategies must implement this method to return a Series of trading signals.

**Signal Convention:**

| Sign | Signal |
| ----------- | ----------- |
|1 | Buy|
|-1| Sell|
|0 | Hold|

---

### `MACrossoverStrategy` — Moving Average Crossover (`ma_crossover_strategy.py`)

**Inputs**:

- X: A DataFrame containing price data, with at least a 'Close' column.

**Logic**:
- Computes short-term and long-term moving averages using rolling windows.
- Generates buy signals when short MA crosses above long MA.
- Generates sell signals when short MA crosses below long MA.


| Parameters | Description |
| ----------- | ----------- |
|short_window (default=20) | Short-term moving average window.|
|long_window (default=50)|Long-term moving average window.|

---

### `SimpleMovingAverageStrategy` — Procedural SMA Logic (`sma_strategy.py`)
```py
class SimpleMovingAverageStrategy:
    def on_bar(self, date, row):
        ...
```
`on_bar(date, row)`: Processes individual rows of price data. Checks if short MA is above or below long MA and triggers appropriate signals.


**Sign Generation**
|Sign| Description|
| ----------- | ----------- |
|buy_signal(date, price)| Returns a dictionary for buy action.|
|sell_signal(date, price)| Returns a dictionary for sell action.|
|get_signals(date, row)| Returns both buy and sell signals.|

**Example Signal Output**
```py
{
    'action': 'BUY',
    'symbol': 'AAPL',
    'quantity': 100,
    'price': 150.23,
    'entry_time': '2024-01-01'
}
```