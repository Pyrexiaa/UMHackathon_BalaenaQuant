from sample_data import BTC_DATA
from quantpilot.backtester import Backtester
<<<<<<< HEAD
from quantpilot.strategy import MLStrategy
=======
from quantpilot.strategy import MLStrategy, BaseStrategy, HMMStrategy, CombinedStrategy
>>>>>>> 3fe2a1d673196adbc0912aa70abf5314abcb2abe
from quantpilot.models import get_model
from quantpilot.visualization import run_dashboard

if __name__ == "__main__":
    
    bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("xgboost")), strategy_name="XGBoost",
                    mode="geometric", entry_exit_logic="mean_reversion")
<<<<<<< HEAD

    bt.run(forward_test=True, forward_start_date="2024-01-01") 

    bt.plot_results()
   
    run_dashboard()
    
=======
    # bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("xgboost")), strategy_name="XGBoost",
    #             mode="geometric", entry_exit_logic="mean_reversion")
    # bt = Backtester(data=BTC_DATA, strategy=HMMStrategy(), strategy_name="HMM",
    #             mode="arithmetic", entry_exit_logic="mean_reversion")
    # bt = Backtester(data=BTC_DATA, 
    #                 strategy=CombinedStrategy(hmm_strategy=HMMStrategy(), 
    #                                         #   ml_strategy=MLStrategy(get_model("tcn")),
    #                                             ml_strategy=MLStrategy(get_model("xgboost")),
    #                                           method="and"
    #                                           ), 
    #                 strategy_name="Combined",
    #             mode="geometric", entry_exit_logic="mean_reversion")
    
    # bt.run(backtest_end_date="2023-12-31") # Run backtest only
    bt.run(forward_test=True, forward_start_date="2024-01-01") # Run backtest and forward test

    bt.plot_results()
   
    # run_dashboard()
    

    
    
# Backtesting with 'target' column

# if __name__ == "__main__":
#     class TargetStrategy(BaseStrategy):
#         def generate_signals(self, X: pd.DataFrame) -> pd.Series:
#             # Generate signals based on target column
#             signal = X['target'].copy()
#             signal = pd.Series(signal, index=X.index)
#             # convert signal 0 1 2 to -1 0 1
#             signal = signal.replace({0: -1, 1: 0, 2: 1})
#             return signal
        
#     # Run backtest and forward test
#     # bt = Backtester(data=BTC_DATA, strategy=TargetStrategy(),
#     #                 mode="arithmetic", entry_exit_logic="trend_following")
#     bt = Backtester(data=BTC_DATA, strategy=TargetStrategy(), strategy_name= "TargetStrategy",
#                 mode="geometric", entry_exit_logic="mean_reversion")
#     bt.run(forward_test=True, forward_start_date="2024-01-01")
#     bt.plot_results()
#     run_dashboard()
>>>>>>> 3fe2a1d673196adbc0912aa70abf5314abcb2abe
