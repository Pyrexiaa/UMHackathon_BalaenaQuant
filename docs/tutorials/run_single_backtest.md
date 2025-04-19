## Run Backtest Only or with Forward Test

This guide shows you how to run **flexible backtests** and **forward tests** using the `Backtester` in QuantPilot.

---

### Prerequisites
- Installed QuantPilot and dependencies ([see setup](https://github.com/Pyrexiaa/UMHackathon_BalaenaQuant/?tab=readme-ov-file#quantpilot))
- A Python 3.8+ environment
- Prepared time series dataset (e.g., `BTC_DATA`)

---

### Step 1: Initialize Strategy and Backtester
```python
from quantpilot.models import get_model
from quantpilot.strategy import MLStrategy
from quantpilot.backtester import Backtester
from sample_data import BTC_DATA  # Your historical price data

bt = Backtester(
    data=BTC_DATA,
    strategy=MLStrategy(get_model("xgboost")),
    strategy_name="XGBoost",
    mode="geometric",
    entry_exit_logic="mean_reversion"
)
```

---

### Step 2: Run Backtest Only (Flexible Options)
```python
# Run full backtest from beginning to end of dataset
tbt.run()

# Run backtest up to a specific end date
bt.run(backtest_end_date="2023-12-31")

# Run backtest within specific date range
bt.run(backtest_start_date="2021-01-01", backtest_end_date="2022-12-21")

# Run backtest for a fixed number of years from start
tbt.run(backtest_years=3)
```

---

### Step 3: Run Backtest + Forward Test (Flexible Options)
```python
# Run full backtest followed by forward test starting from a set date
bt.run(forward_test=True, forward_start_date="2024-01-01")

# Automatically split into backtest and 1-year forward test
bt.run(forward_test=True, forward_years=1)

# Specify forward test start and end dates
bt.run(forward_test=True, forward_start_date="2024-01-01", forward_end_date="2025-01-31")
```

---

### Plot Results
```python
bt.plot_results()
```

---

### Launch Streamlit Dashboard
```python
from quantpilot.visualization.run import run_dashboard
run_dashboard()
```

---

### Example Output
```
STRATEGY NAME: XGBoost
Running backtest phase...
Entry/Exit Logic: Mean Reversion
Mode: Geometric

Running forward test phase...
Entry/Exit Logic: Mean Reversion
Mode: Geometric

============================================================
                 BACKTEST PHASE PERFORMANCE                 
============================================================
Start Date              2020-03-13 02:00:00
End Date                2023-12-31 23:00:00
Duration (days)         1388
Data rows               33183
Initial Capital         $100,000.00
Final Equity            $-343,147.48
Total Return            -443.16%
Annualized Return       nan%
Calmar Ratio            nan
Sharpe Ratio            -1.02
Sortino Ratio           -1.16
Max Drawdown            -428.33%
Number of Trades        28125
Long Trades             7031
Short Trades            7031
Win Rate                0.59
Average PnL             4.62
Expectancy              4.62
Profit Factor           1.05
Average Holding Period  0 days 02:22:13.295406058
Trade Frequency         0.85

============================================================
               FORWARD TEST PHASE PERFORMANCE               
============================================================
Start Date              2024-01-01 00:00:00
End Date                2025-03-31 22:00:00
Duration (days)         455
Data rows               10913
Initial Capital         $100,000.00
Final Equity            $-231,334.59
Total Return            -331.39%
Annualized Return       nan%
Calmar Ratio            nan
Sharpe Ratio            -1.42
Sortino Ratio           -1.76
Max Drawdown            -336.55%
Number of Trades        9201
Long Trades             2300
Short Trades            2300
Win Rate                0.59
Average PnL             14.02
Expectancy              14.02
Profit Factor           1.08
Average Holding Period  0 days 02:22:41.739130434
Trade Frequency         0.84
============================================================
Results saved to output/portfolio_xgboost.csv
Trades saved to output/trade_xgboost.csv
Records saved to output/records_xgboost.csv
```
### Visualization Example
![alt text](image-1.png)
![alt text](image.png)
