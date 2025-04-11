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


    for metric in metrics_call_using_api:
        # Load data using API
        
        await api_loader.load_data(
            metrics=metric,
            window="hour",
            limit=1000,
            save_data=True
        )
        
        
    for metric in data_source["cryptoquant"]["topics"]:
        # Load data using topic
        await custom_loader.load_data(
            start_time=datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
            end_time=datetime(year=2025, month=4, day=1, tzinfo=timezone.utc),
            metrics=metric,
            save_data=True
        )

asyncio.run(main())
