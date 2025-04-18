import asyncio
from datetime import datetime, timezone

from quantpilot.data import DataLoader  
from quantpilot.features import add_hmm_features, add_nlp_sentiment_score

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
    
    df.to_csv('output_data.csv', index=False)

if __name__ == "__main__":
    asyncio.run(main())