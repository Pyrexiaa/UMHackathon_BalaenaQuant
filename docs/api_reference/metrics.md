# Metrics Module Overview
The Metrics class, defined in metrics.py, is responsible for calculating comprehensive performance metrics after a trading strategy is backtested. It inherits from the abstract base class BaseMetrics in base_metrics.py.


## Class Structure

#### Base Class: `BaseMetrics`
- Base class that contains constructor that stores common inputs
- Requires implementation of all_metrics()


#### Deriviced Class: `Metrics(BaseMetrics)`

**Inputs**:

- `equity`: Time series of portfolio values

- `returns`: Time series of daily returns

- `trades`: Trade log (entry/exit, PnL, etc.)

- `risk_free_rate`: For Sharpe ratio

- `trading_fee`: Cost per trade

- `signals`: Trading signals (for frequency)


## Usage
### Calculate and return the specified metrics
`get_metrics(names: list)`: 
</br>


```py
metrics.get_metrics(['Sharpe Ratio', 'Max Drawdown'])
```
- If names is None, it returns all available metrics.
- Invalid names are ignored (only matching keys are returned).

### Add custom metrics
```py
from .metrics import Metrics

class CustomMetrics(Metrics):
    def win_loss_trade_ratio(self):
        """Ratio of winning to losing trades."""
        if self.trades is None or self.trades.empty:
            return None
        wins = len(self.trades[self.trades['pnl'] > 0])
        losses = len(self.trades[self.trades['pnl'] <= 0])
        return wins / losses if losses > 0 else float('inf')

    def custom_all_metrics(self):
        """
        Extend the base all_metrics() with custom metrics.
        """
        base_metrics = super().all_metrics()
        base_metrics.update({
            'Win/Loss Trade Ratio': self.win_loss_trade_ratio(),
        })
        return base_metrics
```


## Output
#### Return & Risk Metrics

|Function	|Description|
|---|---|
|total_return()|	Total portfolio return over the period|
|annualized_return()|	CAGR (Compounded Annual Growth Rate)|
|annualized_volatility()|	Std. dev. of daily returns scaled to annual|
|calmar_ratio()	|Annual return divided by absolute max drawdown|
|sharpe_ratio()|	Measures risk-adjusted return (excess return / volatility)|
|sortino_ratio()|	Like Sharpe, but penalizes only downside volatility|
|max_drawdown()|	Largest peak-to-trough decline in equity|
<br/>

#### Trade-Level Metrics
|Function	| Description|
|--|--|
| Number of Trades |	Returns the number of executed trades|
| Long Trades |	Returns the number of long trades|
| Short Trades |	Returns the number of short trades|
| Win Rate	| Ratio of winning trades|
| Expectancy	|Avg. expected PnL per trade|
| Profit Factor	|Total profits ÷ total losses|
| Avg Holding Period|	Mean duration of holding an asset|
| Trade Frequency	|Number of trades per data row|




