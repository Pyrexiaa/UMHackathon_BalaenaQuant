import pandas as pd
from src.backtester import Backtester
from src.strategy import MLStrategy
from src.models import ModelUtils

# Load data
df = pd.read_csv("sample_data/btc_data.csv", index_col=0, parse_dates=True)

# Load model
model = ModelUtils.get_model("xgboost")

# Initialize strategy
strategy = MLStrategy(
    model=model,
    features=df.columns.tolist(),
    target_column="target",
    threshold=0.5,
)

bt = Backtester(data=df, strategy=strategy)
bt.run()

# Run backtest
bt = Backtester(data=df, strategy=strategy)
bt.run()

# Show metrics
bt.get_performance_metrics()

# work in progress