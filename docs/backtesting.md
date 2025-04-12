# Backtesting Pipeline Overview

The backtesting pipeline simulates trading strategies over historical market data to evaluate their performance before live deployment. This is implemented through the Backtester class in backtester.py

## Key Concepts
**Strategy**: Contains signal generation logic (BUY, HOLD, SELL).

**Portfolio**: Tracks cash, holdings, and total value over time.

**Trades**: Executed whenever a signal changes the position.

**Metrics**: Evaluates performance (returns, Sharpe ratio, drawdowns, etc.).

**Forward Testing**: Tests strategy on unseen future data.

--- 

### Important Functions

`__init__()` :
Initializes the backtester with:

- Market data

- Trading strategy

- Capital & fees

- Risk-free rate for metrics

</br>

`run()`:
Main method to run the backtest.

- Optionally supports forward testing

- Calls _run_phase() for backtest/forward phase

- Computes and stores metrics via _calculate_metrics()

- Generates summary report

</br>

`_run_phase(data)`:
Simulates strategy on given price data.

- Applies strategy signals

- Executes trades (buys/sells)

- Updates portfolio (cash, holdings, equity)

- Computes:

    - Daily returns

    - Cumulative returns

    - Drawdowns via _calculate_drawdown()

</br>

`_calculate_drawdown(equity_curve)`:
Computes drawdown as the percentage drop from historical peak equity:
```py
drawdown = (equity - equity.cummax()) / equity.cummax()
```
</br>

`_calculate_metrics(results, trades)`: 
Delegates to the Metrics class to compute:

- Sharpe Ratio

- Sortino Ratio

- Win Rate

- Max Drawdown
CAGR, etc.

</br>

`generate_report(phase)`:
Displays a performance summary in tabular format using tabulate, for:

- Backtest

- Forward Test

- Combined Results

</br>

`plot_results()`: 
Creates plots for:

- 📈 Equity Curve

- 📉 Drawdowns

- 🔁 Trading Signals

</br>

`export_to_csv()`:
Saves the results and trades as CSV files.

<br/>

**Signal Configuration (`config.py`)**

```py
BUY_SIGNAL = 0
HOLD_SIGNAL = 1
SELL_SIGNAL = 2
FEE_RATE = 0.0006
```