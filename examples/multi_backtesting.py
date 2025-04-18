from quantpilot.strategy import MLStrategy
import pandas as pd
from quantpilot.multibacktester import Multibacktester
from quantpilot.models import get_model

# Load your price data
price_data = pd.read_csv("data.csv", parse_dates=True, index_col="date")

# Run them together
bt = Multibacktester(
    data=price_data,
    strategies=[MLStrategy(get_model("TCN")), MLStrategy(get_model("XGBOOST"))],
    strategy_names=["MLStrategy_TCN", "MLStrategy_XGBoost"]
)

summary = bt.run_all(forward_test=False)

