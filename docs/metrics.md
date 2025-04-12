# Metrics Module Overview
The Metrics class, defined in metrics.py, is responsible for calculating comprehensive performance metrics after a trading strategy is backtested. It inherits from the abstract base class BaseMetrics in base_metrics.py.


## Class Structure

###  Base Class: BaseMetrics (`base_metrics.py`)
- Contains constructor that stores common inputs
- Requires implementation of all_metrics()

<br/>

`Metrics(BaseMetrics)`

Inherits: `BaseMetrics`

**Inputs**:

`equity`: Time series of portfolio values

`returns`: Time series of daily returns

`trades`: Trade log (entry/exit, PnL, etc.)

`risk_free_rate`: For Sharpe ratio

`trading_fee`: Cost per trade

`signals`: Trading signals (for frequency)

### Core Functionalities
#### Return & Risk Metrics

|Function	|Description|
|---|---|
|total_return()|	Total portfolio return over the period|
|annualized_return()|	CAGR (Compounded Annual Growth Rate)|
|annualized_volatility()|	Std. dev. of daily returns scaled to annual|
|sharpe_ratio()|	Measures risk-adjusted return (excess return / volatility)|
|sortino_ratio()|	Like Sharpe, but penalizes only downside volatility|
|max_drawdown()|	Largest peak-to-trough decline in equity|
|calmar_ratio()	|Annual return divided by absolute max drawdown|
|drawdown_duration()|	Longest number of bars equity stayed below peak|

<br/>

#### Trade-Level Metrics
|Function	| Description|
|--|--|
|trade_metrics()|	Returns a dictionary of metrics derived from executed trades|
| Win Rate	| Ratio of winning trades|
| Expectancy	|Avg. expected PnL per trade|
| Profit Factor	|Total profits ÷ total losses|
| Avg Holding Period|	Mean duration of holding an asset|
| Trade Frequency	|Number of trades per day|

#### 📦 Composite Metric Output

`all_metrics()`
Returns a dictionary of:
- Return & risk metrics
- Trade-level metrics


#### ✂️ Selective Access
`get_metrics(names: list)`: 
Fetch a subset of metrics

```py
metrics.get_metrics(['Sharpe Ratio', 'Max Drawdown'])
```
