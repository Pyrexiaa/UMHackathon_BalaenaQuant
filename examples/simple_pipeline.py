import pandas as pd
from src.backtester import Backtester
from src.strategy.ml_strategy import MLStrategy
from src.models.utils import ModelUtils

btc_data = pd.read_csv("sample_data/btc_data.csv", index_col=0, parse_dates=True)

model = ModelUtils.load_model("experimental/modeling/models/xgboost_model.pkl")

strategy = MLStrategy(
    model=model,
    features=btc_data.columns.tolist(),
    target_column="target",
    threshold=0.5,
)

backtester = Backtester(
    data=btc_data,
    strategy=strategy,
)

backtester.run()

metrics = backtester.get_performance_metrics()
print(metrics)