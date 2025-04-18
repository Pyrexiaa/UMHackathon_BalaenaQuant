from sample_data import BTC_DATA
from quantpilot.strategy import MLStrategy
import pandas as pd
from quantpilot.multibacktester import Multibacktester
from quantpilot.models import get_model

# Run multiple strategies together
bt = Multibacktester(
    data=BTC_DATA,
    strategies=[MLStrategy(get_model("TCN")), MLStrategy(get_model("XGBOOST"))],
    strategy_names=["MLStrategy_TCN", "MLStrategy_XGBoost"],
    mode="geometric", 
    entry_exit_logic="mean_reversion"
)

bt.run_all(forward_test=True, forward_start_date="2024-01-01")

