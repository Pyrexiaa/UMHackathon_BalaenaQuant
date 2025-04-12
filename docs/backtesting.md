r# Backtester

A flexible and extensible backtesting engine for evaluating trading strategies using historical data. Supports backtesting, forward testing, performance visualization, and result exporting.


## Quickstart

....
```python
bt = Backtester(data, strategy)
# Run backtest and forward test
bt.run(forward_test=True, forward_start_date="2024-01-01") 
# Run backtest only
bt.run() 

bt.plot_results()
```

### Output
```
============================================================
                 BACKTEST PHASE PERFORMANCE                 
============================================================
Start Date                    2020-03-13
End Date                      2023-12-31
Duration (days)               33276
Initial Capital               $100,000.00
Final Equity                  $268,437.59
Total Return                  168.44%
Annualized Return             34.52%
Sharpe Ratio                  1.94
Sortino Ratio                 2.13
Max Drawdown                  -22.38%
Calmar Ratio                  2.79
Number of Trades              1320
Win Rate                      0.63
Average PnL                   127.88
Expectancy                    102.34
Profit Factor                 2.47
Average Holding Period        2 days 06:45:00
Trade Frequency (trades/day)  8.97

============================================================
               FORWARD TEST PHASE PERFORMANCE               
============================================================
Start Date                    2024-01-01
End Date                      2025-03-31
Duration (days)               456
Initial Capital               $100,000.00
Final Equity                  $129,836.41
Total Return                  29.84%
Annualized Return             22.51%
Sharpe Ratio                  1.82
Sortino Ratio                 1.89
Max Drawdown                  -18.24%
Calmar Ratio                  2.72
Number of Trades              428
Win Rate                      0.59
Average PnL                   8.47
Expectancy                    6.39
Profit Factor                 1.91
Average Holding Period        1 days 21:30:00
Trade Frequency (trades/day)  7.23
============================================================


```



## Key Concepts
**Strategy**: Contains signal generation logic (BUY, HOLD, SELL).

**Portfolio**: Tracks cash, holdings, and total value over time.

**Trades**: Executed whenever a signal changes the position.

**Metrics**: Evaluates performance (returns, Sharpe ratio, drawdowns, etc.).

**Forward Testing**: Tests strategy on unseen future data.


## Initialization

```python
Backtester(
    data: pd.DataFrame,
    strategy,
    initial_capital: float = 100000,
    risk_free_rate: float = 0.0,
    trading_fee: float = 0.0006
)
```

### Parameters:

- `data`: Historical price data as a Pandas DataFrame with a DateTime index. Must contain a `'close'` column.
- `strategy`: A strategy object implementing `generate_signals(data: pd.DataFrame) -> pd.Series`.
- `initial_capital`: Initial portfolio capital. Default is `100000`.
- `risk_free_rate`: Annual risk-free rate for Sharpe ratio calculation. Default is `0.0`.
- `trading_fee`: Fee per trade as a fraction (e.g., `0.0006` for 0.06%). Default is `0.0006`.

---

## 🚀 Run Backtest

### `run()`

```python
run(
    forward_test: bool = False,
    forward_years: Optional[float] = None,
    forward_start_date: Optional[str] = None
) -> pd.DataFrame
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

