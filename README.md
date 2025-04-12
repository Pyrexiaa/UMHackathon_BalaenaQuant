# QuantPilot

## Project Overview
QuantPilot is a high-performance, modular framework tailored for developing, backtesting, and evaluating trading strategies. It is built to support systematic strategy research using both traditional and machine learning-based approaches. With a flexible architecture and seamless integration across modules, QuantPilot empowers researchers and traders to rapidly iterate, test, and deploy alpha-generating ideas with real-world constraints and metrics.

## Getting Started
Prerequisites:
- Python 3.10 or higher
- To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Project Structure
```
QuantPilot/
│
│   ├── docs/                            # Documentation files
│   │   └── user_guide.md                # Detailed guide for using the backtester
│   │ 
│   ├── examples/                        # Example scripts for backtesting strategies
│   │   └── simple_backtesting.py        # Example of a basic backtesting strategy
│   │ 
│   ├── experimental/                    # Experimental features and development
│   │ 
│   ├── src/                             # Source code for backtesting and strategy logic
│   │   ├── data/                        # Code related to data loader and preprocessing
│   │   │   ├── api/
│   │   │   └── loader/
│   │   │
│   │   ├── features/                    # Feature engineering code
│   │   │   ├── base_features.py
│   │   │   ├── feature_selection.py
│   │   │   ├── ml_features.py
│   │   │   ├── technical_indicators.py
│   │   │
│   │   ├── metrics/                     # Metrics for evaluating strategy performance
│   │   │   ├── base_metrics.py
│   │   │   └── metrics.py
│   │   │
│   │   ├── models/                      # Machine learning models for strategy prediction
│   │   │   ├── base_model.py
│   │   │   ├── [name]_model.py
│   │   │   └── utils.py
│   │   │
│   │   ├── strategy/                    # Strategy-specific code and logic
│   │   │   ├── base_strategy.py
│   │   │   └── ml_strategy.py
│   │   │
│   │   ├── backtester.py                # Core backtesting logic (handles execution of trades, backtesting, forward testing)
│   │   └── config.py                    # Configuration file for parameters and settings
│   │
│
├── .env.example                         # Example environment variables file
├── .gitignore                           # Git ignore file
│
├── requirements.txt                     # List of required Python packages
└── README.md                            # Project overview and setup instructions

```

## Usage

```python
from src.backtester import Backtester
from src.strategy import MLStrategy
from src.models import get_model

if __name__ == "__main__":
    bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("TCN")))
    bt.run(forward_test=True, forward_start_date="2024-01-01")
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



## Documentation
[Presentation Slides](https://www.canva.com/design/DAGkV7mcnvA/eN3IcmLJmv-_Hmy_4f3bnA/view?utm_content=DAGkV7mcnvA&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h61e831623c)
Detailed documentation is available in the [docs](https://github.com/Pyrexiaa/UMHackathon_BalaenaQuant/tree/main/docs)



