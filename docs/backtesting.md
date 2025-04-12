# Backtester

A flexible and extensible backtesting engine for evaluating trading strategies using historical data. Supports backtesting, forward testing, performance visualization, and result exporting.


## Key Concepts
**Strategy**: Contains signal generation logic (BUY, HOLD, SELL).

**Portfolio**: Tracks cash, holdings, and total value over time.

**Trades**: Executed whenever a signal changes the position.

**Metrics**: Evaluates performance (returns, Sharpe ratio, drawdowns, etc.).

**Forward Testing**: Tests strategy on unseen future data.


## Initialization

```python
bt = Backtester(data, strategy)
```

### Parameters:

- `data`: Historical price data as a Pandas DataFrame with a DateTime index. Must contain a `'close'` column.
- `strategy`: A strategy object implementing `generate_signals(data: pd.DataFrame) -> pd.Series`.

---

## 🚀 Run Backtest

### `run()`

```python
# Run backtest and forward test
bt.run(forward_test=True, forward_start_date="2024-01-01")

# Run backtest only
bt.run()
```

Run the backtest with optional forward testing.

#### Parameters:
- `forward_test`: Whether to perform forward testing.
- `forward_years`: Number of years to include in the forward test.
- `forward_start_date`: Start date for forward test if specified.

#### Returns:

- A dictionary with:
  - `'results'`: DataFrame of equity curve and performance.
  - `'metrics'`: Dictionary of performance metrics for different phases.

---

## 📊 Visualization

### `plot_results()`

```python
plot_results()
```

Generates:
- Portfolio equity curve
- Drawdown curve
- Buy/sell signals on price chart

---

## 📄 Reporting

### `generate_report()`
```python
generate_report(phase: str = 'all')
```

Prints a performance report 

- `'all'`: All available phases
- `'backtest'`: Only backtest phase
- `'forward'`: Only forward test
- `'full'`: Combined backtest + forward

---

## 📈 Metrics

#### `performance_metrics()`
Returns performance metrics for the given phase.


```python
performance_metrics(phase: str = 'all') -> Dict
```


---

## 💾 Exporting

#### `export_data()`
Exports Results DataFrame to CSV and Trades log to CSV
```python
export_data(results_path='output/backtest_results.csv', trades_path='output/trade_log.csv')
```

#### `export_metrics()`
Exports performance metrics to CSV.
```python
export_metrics(filepath="output/metrics.csv")
```

---


## 📌 Strategy Interface

Your strategy class should implement the following method:

```python
generate_signals(data: pd.DataFrame) -> pd.Series
```

#### Return values:
- `2` → Buy
- `0` → Sell
- `1` → Hold

---

Happy backtesting! 📉📈

