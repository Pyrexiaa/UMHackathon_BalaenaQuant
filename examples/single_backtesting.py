from sample_data import BTC_DATA
from quantpilot.backtester import Backtester
from quantpilot.strategy import MLStrategy, BaseStrategy
from quantpilot.models import get_model
from quantpilot.visualization import run_dashboard
import pandas as pd

if __name__ == "__main__":
    
    # Run backtest and forward test
    # bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("TCN")),
    #                 mode="arithmetic", entry_exit_logic="mean_reversion")
<<<<<<< HEAD:examples/simple_backtesting.py
    bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("xgboost")),
                mode="arithmetic", entry_exit_logic="trend_following")
=======
    bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("XGBOOST")),
                mode="geometric", entry_exit_logic="mean_reversion")
>>>>>>> 3d22e32849074ea14bba02ad3ff96231fcc23149:examples/single_backtesting.py
    bt.run(forward_test=True, forward_start_date="2024-01-01")
  
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
#     bt = Backtester(data=BTC_DATA, strategy=TargetStrategy(),
#                 mode="geometric", entry_exit_logic="mean_reversion")
#     bt.run(forward_test=True, forward_start_date="2024-01-01")
#     bt.plot_results()