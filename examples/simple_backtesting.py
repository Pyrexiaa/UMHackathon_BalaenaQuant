from sample_data import BTC_DATA
from quantpilot.backtester import Backtester
from quantpilot.strategy import MLStrategy
from quantpilot.models import get_model
from quantpilot.visualization import run_dashboard

if __name__ == "__main__":
    
    # Run backtest and forward test
    bt = Backtester(data=BTC_DATA, strategy=MLStrategy(get_model("TCN")))
    bt.run(forward_test=True, forward_start_date="2024-01-01")

    # Plot results
    bt.export_data()  
    
    # Run dashboard
    run_dashboard()

    
    