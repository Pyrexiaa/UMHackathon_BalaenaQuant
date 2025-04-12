import asyncio
from datetime import datetime, timezone
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from data.loaders.api_loader import APILoader
from data.loaders.custom_loader import CustomLoader
from data.constant.data_source import data_source



async def main():
    custom_loader = CustomLoader(datasource_key="cryptoquant")
    api_loader = APILoader(datasource_key="cryptoquant")
    metrics_call_using_api = ["reserve", "coinbase_premium_index", "taker-buy-sell-stats", "liquidations"]

    # full param
    await custom_loader.run(
        metrics=["difficulty", "estimated_leverage_ratio", "addresses_count", "exchange_whale_ratio", "coinbase_premium_index", "taker-buy-sell-stats", "blockreward", "fees_transaction", "miner_supply_ratio", "addresses_count_inflow", "exchange_supply_ratio", "transactions_count_inflow", "tokens_transferred", "liquidations", "price-ohlcv", "open_interest"],
        start_time=datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
        end_time=datetime(year=2020, month=4, day=1, tzinfo=timezone.utc),
        save_data=True,
        merged=True
    )
    
    # optional param
    # await custom_loader.run(
    #     metrics=["difficulty", "estimated_leverage_ratio", "addresses_count", "exchange_whale_ratio", "coinbase_premium_index", "taker-buy-sell-stats", "blockreward", "fees_transaction", "miner_supply_ratio", "addresses_count_inflow", "exchange_supply_ratio", "transactions_count_inflow", "tokens_transferred", "liquidations"],
    #     save_data=False,
    #     merged=True
    # )
    
    # full param
    # await api_loader.run(
    #     metrics=metrics_call_using_api,
    #     window="hour",
    #     limit=1000,
    #     save_data=True,
    #     merged=True
    # )
    
    # optional param
    # await api_loader.run(
    #     metrics=metrics_call_using_api,
    #     save_data=True
    # )
    
    
        

   
        
        
   

asyncio.run(main())
