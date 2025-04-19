from sample_data import BTC_DATA
from quantpilot.backtester import Backtester
from quantpilot.strategy import MLStrategy
from quantpilot.models import get_model
from quantpilot.visualization import run_dashboard

if __name__ == "__main__":
    
    bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("xgboost")), strategy_name="XGBoost",
                    mode="geometric", entry_exit_logic="mean_reversion", threshold=0.35)

    bt.run(forward_test=True, forward_start_date="2024-01-01") 

    bt.plot_results()
   
    run_dashboard()
    