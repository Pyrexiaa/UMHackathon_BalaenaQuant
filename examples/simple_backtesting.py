from sample_data import BTC_DATA
from quantpilot.backtester import Backtester
from quantpilot.strategy import MLStrategy
from quantpilot.models import get_model
from quantpilot.strategy.ma_crossover_strategy import MACrossoverStrategy
from quantpilot.strategy.base_strategy import BaseStrategy
from quantpilot.visualization import run_dashboard

if __name__ == "__main__":
    
    # Run backtest and forward test
    bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("XGBOOST")))
    # bt = Backtester(data=BTC_DATA, strategy=MACrossoverStrategy(short_window=5, long_window=13))
    bt.run(forward_test=True, forward_start_date="2024-01-01")

    # Plot results
    bt.export_data()  # Save results and trades to file
    run_dashboard()

    
    