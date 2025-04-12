import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.backtester import Backtester
from src.strategy import MLStrategy
from src.models import get_model

if __name__ == "__main__":
    
    file_path = os.path.join(os.path.dirname(__file__), 'sample_data', 'btc_data_with_target_modified.csv')
    
    # Load data
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Load model
    model = get_model("TCN")

    # Initialize strategy
    strategy = MLStrategy(model=model)

    # Run backtest
    bt = Backtester(data=df, strategy=strategy)
    bt.run(forward_test=True, forward_start_date="2024-01-01")
    bt.export_to_csv("my_results.csv", "my_trades.csv")
    bt.plot_results()