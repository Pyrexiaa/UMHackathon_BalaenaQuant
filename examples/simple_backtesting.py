from sample_data import BTC_DATA
from quantpilot.backtester import Backtester
from quantpilot.strategy import MLStrategy
from quantpilot.models import get_model
from quantpilot.strategy.ma_crossover_strategy import MACrossoverStrategy
from quantpilot.strategy.base_strategy import BaseStrategy
from quantpilot.visualization import run_dashboard

if __name__ == "__main__":
    
    # Run backtest and forward test
    # bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("TCN")),
    #                 mode="arithmetic", entry_exit_logic="mean_reversion")
    bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("TCN")),
                mode="arithmetic", entry_exit_logic="trend_following")
    # bt = Backtester(data=BTC_DATA, strategy=MACrossoverStrategy(short_window=5, long_window=13))
    bt.run(forward_test=True, forward_start_date="2024-01-01")
  
    bt.export_data()  # Save results and trades to file
    bt.plot_results()
    # run_dashboard()

    
# # Backtesting with 'target' column
# import pandas as pd
# if __name__ == "__main__":
#     class TargetStrategy(BaseStrategy):
#         def generate_signals(self, X: pd.DataFrame) -> pd.Series:
#             # Generate signals based on target column
#             signal = X['positions'].copy()
#             signal = pd.Series(signal, index=X.index)
#             # convert signal 0 1 2 to -1 0 1
#             signal = signal.replace({0: -1, 1: 0, 2: 1})
#             return signal
        
#     # Run backtest and forward test
#     # bt = Backtester(data=BTC_DATA, strategy=TargetStrategy(),
#     #                 mode="arithmetic", entry_exit_logic="trend_following")
#     bt = Backtester(data=BTC_DATA, strategy=TargetStrategy(),
#                 mode="arithmetic", entry_exit_logic="mean_reversion")
#     bt.run(forward_test=True, forward_start_date="2024-01-01")
#     bt.export_data()  
#     bt.plot_results()