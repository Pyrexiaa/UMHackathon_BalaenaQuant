import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import asyncio
from datetime import datetime, timezone
from src.data import DataLoader  
from src.features import FeaturePipeline 
from src.features.all_features import *

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
    
    pipeline = FeaturePipeline([
        SMA(windows=[50, 200]),
        EMA(),
        RSI(windows=[14]),
        OBV(),
        RSIObvSignal(rsi_window=14),
        MACD(),
        PriceChange(),
        Volatility(),
        BollingerBands(),
        HMM(),
        RollingKMeans(),
        NLPSentiment()
    ])
    new_df = pipeline.add_features(df)
    new_df.to_csv('output_data2.csv', index=False)

if __name__ == "__main__":
    asyncio.run(main())