import asyncio
from datetime import datetime, timezone
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from src.data import APILoader, CustomLoader

async def load_multiple_metric_with_custom_loader():
    """
    Demonstrates how to use CustomLoader with full parameters: 
    metrics, start_time, end_time, save_data, and merged.
    """
    custom_loader = CustomLoader(datasource_key="cryptoquant")
    await custom_loader.run(
        metrics=[
            "difficulty", "estimated_leverage_ratio", "addresses_count", "exchange_whale_ratio",
            "coinbase_premium_index", "taker-buy-sell-stats", "blockreward", "fees_transaction",
            "miner_supply_ratio", "addresses_count_inflow", "exchange_supply_ratio", "transactions_count_inflow",
            "tokens_transferred", "liquidations", "price-ohlcv", "open_interest"
        ],
        start_time=datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
        end_time=datetime(year=2025, month=4, day=1, tzinfo=timezone.utc),
        save_data=True,
        merged=True
    )
    print("CustomLoader with full parameters executed.")


async def load_single_metric_with_custom_loader():
    """
    Demonstrates how to use CustomLoader with optional parameters: 
    metrics, save_data, and merged. The start_time and end_time 
    are not specified here.
    """
    custom_loader = CustomLoader(datasource_key="cryptoquant")
    await custom_loader.run(
        metrics=["price-ohlcv"],
        save_data=False,
        merged=True
    )
    print("CustomLoader with optional parameters executed.")


async def load_multiple_metrics_with_api_loader():
    """
    Demonstrates how to use APILoader with full parameters: 
    metrics, window, limit, save_data, and merged.
    """
    api_loader = APILoader(datasource_key="cryptoquant")
    await api_loader.run(
        metrics=["reserve", "coinbase_premium_index", "taker-buy-sell-stats", "liquidations"],
        window="hour",  # Specify the window for data aggregation
        limit=1000,  # Set the limit for the number of records to fetch
        save_data=True,
        merged=True
    )
    print("APILoader with full parameters executed.")


async def load_single_metric_with_api_loader():
    """
    Demonstrates how to use APILoader with optional parameters: 
    metrics and save_data. The window and limit will use default values.
    """
    api_loader = APILoader(datasource_key="cryptoquant")
    await api_loader.run(
        metrics=["taker-buy-sell-stats"],
        save_data=True
    )
    print("APILoader with optional parameters executed.")


async def main():
    """
    Main function that demonstrates the usage of different loaders 
    (CustomLoader and APILoader) with various parameter configurations.
    """
    # Demonstrate loading multiple metrics with CustomLoader (full parameters)
    await load_multiple_metric_with_custom_loader()

    # Demonstrate loading a single metric with CustomLoader (optional parameters)
    await load_single_metric_with_custom_loader()

    # Demonstrate loading multiple metrics with APILoader (full parameters)
    await load_multiple_metrics_with_api_loader()

    # Demonstrate loading a single metric with APILoader (optional parameters)
    await load_single_metric_with_api_loader()


# Run the main function asynchronously
asyncio.run(main())