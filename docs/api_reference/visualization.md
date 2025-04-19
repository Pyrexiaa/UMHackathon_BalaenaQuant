# Visualization Module
The `quantpilot.visualization module` provides a Streamlit-powered dashboard to explore the results of single or multistrategy backtests. It supports visualizing equity curves, drawdowns, trading signals, and comparison heatmaps across strategies using metadata and portfolio output.

## Key Features
- Automatic detection of available backtests via metadata files (meta_*.json)

- Tabs for each strategy, showing:

- Equity curve (Backtest & Forward test)

- Drawdown (Backtest & Forward test)

- Trading signals overlaid on price

- Metric comparison heatmap across multiple strategies (e.g., Sharpe Ratio)

- Optional cleanup of metadata files upon dashboard exit

## How It Works
1. Each strategy run (via Backtester or Multibacktester) creates:
- A portfolio_<name>.csv file (equity, drawdown, etc.)
- A meta_<name>.json file with backtest & forward test date ranges, and strategy metrics.

2. When run_dashboard() is called:
- All metadata files are loaded and linked to their corresponding portfolio CSVs.
- Each strategy is rendered in its own tab in the Streamlit UI.

3. If multiple strategies are detected:
- A heatmap comparison tool is enabled to visually compare metrics like Sharpe Ratio, Sortino, etc.

## Generation of metadata for each strategy run
```py
periods_metadata = {
            key: {
                'start': str(val['start']),
                'end': str(val['end'])
            } for key, val in self.periods.items()
        }

        # Add metrics to metadata
        periods_metadata["metrics"] = output["metrics"]


        with open(f'output/meta_{self.strategy_name.lower()}.json', 'w') as f:
            json.dump(self.convert_json_friendly(periods_metadata), f, indent=4)
```

## Example Usage
```py
from quantpilot.backtest.multibacktester import Multibacktester
from quantpilot.strategy.ml import MLStrategy
from quantpilot.models import get_model
from quantpilot.visualization.run import run_dashboard

bt = Multibacktester(
    data=BTC_DATA,
    strategies=[
        MLStrategy(get_model("TCN")),
        MLStrategy(get_model("XGBoost"))
    ],
    strategy_names=["TCN", "XGBoost"],
    mode="geometric", 
    entry_exit_logic="mean_reversion"
)

bt.run_all(forward_test=True, forward_start_date="2024-01-01")

# Launch Streamlit dashboard in browser
run_dashboard()
```
## Cleanup Behaviour
When run_dashboard() finishes:

- All `meta_*.json` files are deleted using cleanup_meta_files() (registered with atexit).

- If you want to retain these files, you can comment out or remove the `atexit.register(cleanup_meta_files)` line.