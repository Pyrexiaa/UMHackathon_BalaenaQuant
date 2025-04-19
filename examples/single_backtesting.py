from sample_data import BTC_DATA
from quantpilot.backtester import Backtester
from quantpilot.strategy import MLStrategy, BaseStrategy, HMMStrategy, CombinedStrategy
from quantpilot.models import get_model
from quantpilot.visualization import run_dashboard

if __name__ == "__main__":
    
    # bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("xgboost")), strategy_name="XGBoost",
    #             mode="geometric", entry_exit_logic="mean_reversion", threshold=0.35)
    # bt = Backtester(data=BTC_DATA, strategy=HMMStrategy(), strategy_name="HMM",
    #             mode="geometric", entry_exit_logic="mean_reversion")
    bt = Backtester(data=BTC_DATA, 
                    strategy=CombinedStrategy(hmm_strategy=HMMStrategy(), 
                                              ml_strategy=MLStrategy(get_model("xgboost")),
                                              method="vote"
                                              ), 
                    strategy_name="Combined",
                mode="geometric", entry_exit_logic="mean_reversion", threshold=0.35)
    
    # bt.run(backtest_end_date="2023-12-31") # Run backtest only
    bt.run(forward_test=True, forward_start_date="2024-01-01") # Run backtest and forward test

    bt.plot_results()
   
    # run_dashboard()