# QuantPilot

## Project Overview
QuantPilot is a high-performance, modular framework tailored for developing, backtesting, and evaluating trading strategies. It is built to support systematic strategy research using both traditional and machine learning-based approaches. With a flexible architecture and seamless integration across modules, QuantPilot empowers researchers and traders to rapidly iterate, test, and deploy alpha-generating ideas with real-world constraints and metrics.

## Getting Started
### Prerequisites
- Python 3.10
- Required dependencies (can be installed via requirements.txt -> `pip install -r requirments.txt`)


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
│   │   ├── features/                    # Feature engineering code
│   │   │   ├── base_features.py
│   │   │   ├── feature_selection.py
│   │   │   ├── ml_features.py
│   │   │   ├── technical_indicators.py
│   │   ├── metrics/                     # Metrics for evaluating strategy performance
│   │   │   ├── base_metrics.py
│   │   │   └── metrics.py
│   │   ├── models/                      # Machine learning models for strategy prediction
│   │   │   ├── base_model.py
│   │   │   ├── [name]_model.py
│   │   │   └── utils.py
│   │   ├── strategy/                    # Strategy-specific code and logic
│   │   │   ├── base_strategy.py
│   │   │   └── ml_strategy.py
│   │   ├── backtester.py                # Core backtesting logic (handles execution of trades, backtesting, forward testing)
│   │   └── config.py                    # Configuration file for parameters and settings

```

## Usage

```python
from src.backtester import Backtester
from src.strategy import MLStrategy
from src.models import get_model

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), 'sample_data', 'btc_data.csv')
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    model = get_model("TCN")

    strategy = MLStrategy(model=model)

    bt = Backtester(data=df, strategy=strategy)
    bt.run(forward_test=True, forward_start_date="2024-01-01")
    bt.plot_results()
```


## Presentation Slides
[Slides Link]([https://www.python.org/downloads/](https://www.canva.com/design/DAGkV7mcnvA/eN3IcmLJmv-_Hmy_4f3bnA/view?utm_content=DAGkV7mcnvA&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h61e831623c))

## Framework Architecture Diagram 



