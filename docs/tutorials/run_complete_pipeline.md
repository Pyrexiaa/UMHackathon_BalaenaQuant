## Run Full Backtest with Data Ingestion, Feature Engineering, and Dashboard

### Prerequisites
Ensure you have the following ready:
- Installed **QuantPilot** and all dependencies (see [README.md](https://github.com/Pyrexiaa/UMHackathon_BalaenaQuant/?tab=readme-ov-file#quantpilot))
- Python 3.8+ environment with `asyncio` support

---

### What This Script Does

This example shows how to:
1. Ingest historical on-chain and market data using the `DataLoader`
2. Engineer HMM-based regime features and NLP sentiment scores
3. Run a backtest using a Temporal Convolutional Network (TCN) model
4. Plot performance metrics
5. Launch the Streamlit dashboard for interactive analysis

---

### ⚙️ Step-by-Step Guide

```python
from quantpilot.data import DataLoader
from quantpilot.models import get_model
from quantpilot.strategy import MLStrategy
from quantpilot.features import add_hmm_features, add_nlp_sentiment_score
from quantpilot.backtester import Backtester
from quantpilot.visualization.run import run_dashboard
from datetime import datetime, timezone
import asyncio

async def main():
    # Step 1: Load historical metrics
    loader = DataLoader()
    metrics = [
        "price-ohlcv", "taker-buy-sell-stats", "addresses_count_inflow", "addresses_count",
        "coinbase_premium_index", "estimated_leverage_ratio", "exchange_whale_ratio",
        "tokens_transferred", "blockreward", "fees_transaction", "miner_supply_ratio", 
        "exchange_supply_ratio", "transactions_count_inflow", "liquidations", "open_interest", "difficulty"
    ]
    start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2025, 4, 1, tzinfo=timezone.utc)
    df = await loader.run(metrics=metrics, start_time=start_time, end_time=end_time)

    # Step 2: Feature engineering
    df = add_hmm_features(df)
    df = add_nlp_sentiment_score(df)

    # Step 3: Initialize and run backtest
    bt = Backtester(data=df, strategy=MLStrategy(get_model("TCN")), strategy_name="TCN")
    bt.run(forward_test=True, forward_start_date="2024-01-01")
    bt.plot_results()

    # Step 4: Launch dashboard
    run_dashboard()

# Trigger the main function
if __name__ == "__main__":
    asyncio.run(main())
```

### Final Outcome
```
Fetching data...
Merging data...
CSV saved to output/merged_cryptoquant_1745099231.csv
STRATEGY NAME: TCN
Running backtest phase...
torch.Size([33179, 3])
Entry/Exit Logic: Trend Following
Mode: Arithmetic

Running forward test phase...
torch.Size([10820, 3])
Entry/Exit Logic: Trend Following
Mode: Arithmetic

============================================================
                 BACKTEST PHASE PERFORMANCE                 
============================================================
Start Date              2020-03-13 02:00:00
End Date                2023-12-31 23:00:00
Duration (days)         1388
Data rows               33299
Total Return            72.95%
Annualized Return       15.50%
Calmar Ratio            0.07
Sharpe Ratio            0.06
Sortino Ratio           0.07
Max Drawdown            -229.57%
Number of Trades        4077
Long Trades             1019
Short Trades            1019
Win Rate                0.51
Average PnL             0.00
Expectancy              0.00
Profit Factor           1.20
Average Holding Period  0 days 16:16:16.840039254
Trade Frequency         0.12

============================================================
               FORWARD TEST PHASE PERFORMANCE               
============================================================
Start Date              2024-01-01 00:00:00
End Date                2025-03-31 23:00:00
Duration (days)         455
Data rows               10940
Total Return            47.20%
Annualized Return       36.37%
Calmar Ratio            0.89
Sharpe Ratio            0.15
Sortino Ratio           0.19
Max Drawdown            -40.87%
Number of Trades        245
Long Trades             61
Short Trades            61
Win Rate                0.50
Average PnL             0.01
Expectancy              0.01
Profit Factor           1.39
Average Holding Period  3 days 16:17:12.786885245
Trade Frequency         0.02
============================================================
Results saved to output/portfolio_tcn.csv
Trades saved to output/trade_tcn.csv
Records saved to output/records_tcn.csv
```