import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from examples.sample_data import BTC_DATA
from src.backtester import Backtester
from src.strategy import MLStrategy
from src.models import get_model
import pandas as pd
from src.strategy.ma_crossover_strategy import MACrossoverStrategy
from src.strategy.base_strategy import BaseStrategy

if __name__ == "__main__":
    
    # Run backtest and forward test
    bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("TCN")))
    # bt = Backtester(data=BTC_DATA, strategy=MACrossoverStrategy(short_window=5, long_window=13))
    bt.run(forward_test=True, forward_start_date="2024-01-01")

    # Plot results
    bt.export_data()  # Save results and trades to file
    os.system("streamlit run src/visualization/visualize_streamlit.py")

    
    