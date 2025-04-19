import asyncio
from datetime import datetime, timezone
from quantpilot.backtester import Backtester
from quantpilot.data import DataLoader  
from quantpilot.strategy import MLStrategy
from quantpilot.models import get_model
from quantpilot.features import add_hmm_features, add_nlp_sentiment_score
from quantpilot.visualization.run import run_dashboard

async def main():
    loader = DataLoader()
    metrics = [
        "price-ohlcv", "taker-buy-sell-stats", "addresses_count_inflow", "addresses_count",
        "coinbase_premium_index", "estimated_leverage_ratio", "exchange_whale_ratio",
        "tokens_transferred", "blockreward", "fees_transaction", "miner_supply_ratio", 
        "exchange_supply_ratio", "transactions_count_inflow", "liquidations", "open_interest", "difficulty"
    ]
    start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)  
    end_time = datetime(2025, 4, 1, tzinfo=timezone.utc)   
    df = await loader.run(metrics=metrics, start_time=start_time, end_time=end_time) 
    
    df = add_hmm_features(df)
    df = add_nlp_sentiment_score(df)
    
    bt = Backtester(data=df, strategy=MLStrategy(get_model("TCN")), strategy_name="TCN")
    bt.run(forward_test=True, forward_start_date="2024-01-01")
    bt.plot_results()
    run_dashboard()

if __name__ == "__main__":
    asyncio.run(main())