# 📘 Strategy Module Documentation

This module provides a structured and extensible way to implement and test trading strategies using a backtesting framework. It includes a base abstract class and two example strategy implementations based on moving averages.


## 🤖 MLStrategy – Machine Learning-Based Trading Strategy

The `MLStrategy` class extends the `BaseStrategy` and utilizes a trained machine learning model to generate buy, hold, or sell trading signals based on predicted probabilities.



## 🧠 `MLStrategy`

A machine learning-driven trading strategy that converts model predictions into actionable trading signals.

---

### 🔧 Constructor

```py
MLStrategy(model: Any, 
           buy_threshold: float = Config.BUY_THRESHOLD, 
           sell_threshold: float = Config.SELL_THRESHOLD)
```

|Parameter	|Type|	Description|
| ----------- | ----------- |----|
|model|	Any|	A trained ML model that exposes a .predict() method returning class probabilities|
|buy_threshold|	float|	Minimum probability threshold to generate a Buy signal|
|sell_threshold|	float|	Minimum probability threshold to generate a Sell signal|


#### Method: `generate_signals()`
    `generate_signals(X: pd.DataFrame) -> pd.Series`

|Parameter|	Type|	Description|
| ----------- | ----------- |----|
|X|	pd.DataFrame|	Feature matrix with a DateTimeIndex|

Returns: 
A pandas.Series of integer trading signals:

|Signal|	Action|	
| ----------- | ----------- 
|2 | Buy|
|1 | Hold|
|0 | Sell|



**Logic**

    1. Calls model.predict(X) to retrieve class probabilities.

    2. Checks for shape (must be Nx3 for 3-class classification).

    3. Applies thresholds:

        If Buy probability > buy_threshold → Buy (2)

        If Sell probability > sell_threshold → Sell (0)

        Else → Hold (1)

    4. Handles prediction alignment:

    5. Skips early rows based on the model’s window_size (or Config.WINDOW_SIZE).

    6. Pads warm-up period with Hold signals.

    7. Returns signal series aligned with the input index.

**Validation**

- Checks output probability shape (Nx3)

- Ensures probabilities across each row sum to 1

- Handles short prediction sequences with fallback Hold signals

- Raises informative errors if prediction fails

**Example Usage**
```py
model = MyTrainedModel()  # Must implement .predict() returning probs of shape (N, 3)
strategy = MLStrategy(model=model, buy_threshold=0.6, sell_threshold=0.6)

signals = strategy.generate_signals(X=feature_dataframe)
```
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