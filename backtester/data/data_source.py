from dotenv import load_dotenv
import os

load_dotenv()
data_source = {
    "base_url": "https://api.datasource.cybotrade.rs",
    "api_key": os.getenv("CYBOTRADE_API_KEY"),
    "cryptoquant": {
        
        "endpoints": {
            "reserve": "/cryptoquant/btc/exchange-flows/reserve?exchange=binance&window=day&limit=2",
            "coinbase_premium_index": "/cryptoquant/btc/market-data/coinbase-premium-index?window=hour&limit=10",
            
        }
    },
    "glassnode": {
        
        "endpoints": {
            
        }
    },

    "coinbase": {
       
        "endpoints": {
            
        }
    }
}

feature_topic_dict = {
    "difficulty": "cryptoquant|btc/network-data/difficulty?window=hour",
    "estimated_leverage_ratio": "cryptoquant|btc/market-indicator/estimated-leverage-ratio?exchange=binance&window=hour",
    "addresses_count": "cryptoquant|btc/network-data/addresses-count?window=hour",
    "exchange_whale_ratio": "cryptoquant|btc/flow-indicator/exchange-whale-ratio?exchange=binance&window=hour",
    "coinbase_premium_index": "cryptoquant|btc/market-data/coinbase-premium-index?window=hour",
    "coinbase_premium_gap_usdt_adjusted": "",
    "coinbase_premium_index_usdt_adjusted": "",
    "taker-buy-sell-stats": "cryptoquant|btc/market-data/taker-buy-sell-stats?window=hour",
    "blockreward": "cryptoquant|btc/network-data/blockreward?window=hour",
    "fees_transaction": "cryptoquant|btc/network-data/fees-transaction?window=hour",
    "miner_supply_ratio": "cryptoquant|btc/flow-indicator/miner-supply-ratio?miner=f2pool&window=hour",
    "addresses_count_inflow": "",
    "addresses_count_outflow": "",
    "exchange_supply_ratio": "cryptoquant|btc/flow-indicator/exchange-supply-ratio?exchange=binance&window=hour",
    "transactions_count_inflow": "",
    "transactions_count_outflow": "",
    "tokens_transferred": "cryptoquant|btc/network-data/tokens-transferred?window=hour",
    "transactions_count_inflow.1": "",
    "transactions_count_outflow.1": "",
    "liquidations": "cryptoquant|btc/market-data/liquidations?window=hour",
    "price-ohlcv": "cryptoquant|btc/market-data/price-ohlcv?window=hour",
    "open_interest": "cryptoquant|btc/market-data/open-interest?window=hour"
}

