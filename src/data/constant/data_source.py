from dotenv import load_dotenv
import os

load_dotenv()

data_source = {
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