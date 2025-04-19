# Data Ingestion
The <LIBRARY_NAME> ingestion module enables users to fetch and preprocess data from various providers using either API-based clients or custom topic loaders. It leverages the cybotrade-resource package to streamline the data collection process for alpha research, analysis, and strategy development.

## Loaders
### Base Loader `base_loader.py`
The BaseLoader class provides a general interface for data loaders. Subclasses must implement the load_data() and run() methods to define their own ingestion logic.

- `load_data()`- [abstract]: To be overridden in subclasses. Defines how to fetch data from a data source.
- `clean_data()`: Removes duplicate rows and handles missing values in the data.
- `get_data()`: Returns the currently stored data as a `pandas.DataFrame`.
- `reset_data()`: Clears the loader data storage (DataFrame).
- `save_data_to_csv()`: Exports the stored data to a CSV file.
- `merge_csv()`: Merges multiple metric CSVs into a single DataFrame and exports it.
- `run()`: Defines the end-to-end pipeline: `load_data() → clean_data() → get_data()`

### API Loader `api_loader.py`
A loader that extracts data via HTTP API requests using the APIClient
#### load_data()
```py
- async def load_data(self,
        metrics: list[str],
        window: Optional[str] = "hour",
        limit: Optional[int] = 46211,
        save_data: Optional[bool] = True,
        merged: Optional[bool] = False
    ): 
```

| Paramaters | Description |
| ----------- | ----------- |
| metrics: list[str] | list of metrics to be extracted from data source |
| window: Optional[str] | interval between the timestamp of each data row. Defaults to `hour` if omitted |
|limit: Optional[int]| The number of items to return (maximum: 100000). Defaults to `46211` if omitted|
|save_data: Optional[bool]| To save extracted data frame of single metric and export to csv file. Defaults to `True` if omitted|
|merged: Optional[bool]| To merge all extracted metrics into single DataFrame and export to csv file. Defaults to `False` if omitted|


#### API client `api_client.py`
To send API requests to different data sources / provider
- get() -> DataFrame: send GET requests to datasource.

#### ClientConfig `client_config.py`
Initialize API key and base url for API client.
- get_api_key(): Fetch the API key for a specific source.
- get_base_url(): Fetch the base URL for a given data provider.

### Topic Loader
A loader that retrieves data through the cybotrade-resource package using topic strings.
#### load_data()
```py
 async def load_data(
        self,
        metrics: list[str],
        start_time: Optional[datetime] = datetime(
            year=2020, month=1, day=1, tzinfo=timezone.utc
        ),
        end_time: Optional[datetime] = datetime(
            year=2025, month=4, day=1, tzinfo=timezone.utc
        ),
        save_data: Optional[bool] = True,
        merged: Optional[bool] = False
    ): 
```

| Paramaters | Description |
| ----------- | ----------- |
| metrics: list[str] | list of metrics to be extracted from data source |
| start_time: Optional[datetime] |The beginning time of the data (UNIX timestamp in milliseconds) |
|end_time: Optional[datetime]| The end time of the data (UNIX timestamp in milliseconds)|
|save_data: Optional[bool]| To save extracted data frame of single metric and export to csv file. Defaults to `True` if omitted|
|merged: Optional[bool]| To merge all extracted metrics into single DataFrame and export to csv file. Defaults to `False` if omitted|


### Constant `data_source.py`
Dictionary that stores metrics - topics/endpoint key value pair to pass into loaders
```py
datasource = {
    "cryptoquant": {
        "base_url": "https://api.datasource.cybotrade.rs",
        "api_key": os.getenv("CYBOTRADE_API_KEY"),
        "endpoints": {
            "reserve": "/cryptoquant/btc/exchange-flows/reserve?exchange=binance",
            "coinbase_premium_index": "/cryptoquant/btc/market-data/coinbase-premium-index",
            "taker-buy-sell-stats": "/cryptoquant/btc/market-data/taker-buy-sell-stats?exchange=binance",
            "liquidations": "/cryptoquant/btc/market-data/liquidations?exchange=binance",
        },
        "topics": {
            "difficulty": "cryptoquant|btc/network-data/difficulty?window=hour",
            "estimated_leverage_ratio": "cryptoquant|btc/market-indicator/estimated-leverage-ratio?exchange=binance&window=hour",
            "addresses_count": "cryptoquant|btc/network-data/addresses-count?window=hour",
            "exchange_whale_ratio": "cryptoquant|btc/flow-indicator/exchange-whale-ratio?exchange=binance&window=hour",
            "coinbase_premium_index": "cryptoquant|btc/market-data/coinbase-premium-index?window=hour",
            "taker-buy-sell-stats": "cryptoquant|btc/market-data/taker-buy-sell-stats?window=hour&exchange=binance",
            "blockreward": "cryptoquant|btc/network-data/blockreward?window=hour",
            "fees_transaction": "cryptoquant|btc/network-data/fees-transaction?window=hour",
            "miner_supply_ratio": "cryptoquant|btc/flow-indicator/miner-supply-ratio?miner=f2pool&window=hour",
            "addresses_count_inflow": "cryptoquant|btc/exchange-flows/addresses-count?exchange=binance&window=hour",
            "exchange_supply_ratio": "cryptoquant|btc/flow-indicator/exchange-supply-ratio?exchange=binance&window=hour",
            "transactions_count_inflow": "cryptoquant|btc/exchange-flows/transactions-count?exchange=binance&window=hour",
            "tokens_transferred": "cryptoquant|btc/network-data/tokens-transferred?window=hour",
            "liquidations": "cryptoquant|btc/market-data/liquidations?window=hour&exchange=binance",
            "price-ohlcv": "cryptoquant|btc/market-data/price-ohlcv?window=hour",
            "open_interest": "cryptoquant|btc/market-data/open-interest?window=hour&exchange=binance",
        },
    },
    "glassnode": {"endpoints": {}},
    "coinbase": {"endpoints": {}},
}
```