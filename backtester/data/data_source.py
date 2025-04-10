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
    "time": "",
    "difficulty": "",
    "estimated_leverage_ratio": "cryptoquant|btc/market-indicator/estimated-leverage-ratio?exchange=binance&window=day",
    "addresses_count_active": "",
    "addresses_count_sender": "",
    "addresses_count_receiver": "",
    "exchange_whale_ratio": "",
    "coinbase_premium_gap": "",
    "coinbase_premium_index": "",
    "coinbase_premium_gap_usdt_adjusted": "",
    "coinbase_premium_index_usdt_adjusted": "",
    "taker_buy_volume": "",
    "taker_sell_volume": "",
    "taker_buy_ratio": "",
    "taker_sell_ratio": "",
    "taker_buy_sell_ratio": "",
    "blockreward": "",
    "blockreward_usd": "",
    "fees_transaction_mean": "",
    "fees_transaction_mean_usd": "",
    "fees_transaction_median": "",
    "fees_transaction_median_usd": "",
    "miner_supply_ratio": "",
    "addresses_count_inflow": "",
    "addresses_count_outflow": "",
    "exchange_supply_ratio": "",
    "transactions_count_inflow": "",
    "transactions_count_outflow": "",
    "tokens_transferred_total": "",
    "tokens_transferred_mean": "",
    "tokens_transferred_median": "",
    "transactions_count_inflow.1": "",
    "transactions_count_outflow.1": "",
    "long_liquidations": "",
    "short_liquidations": "",
    "long_liquidations_usd": "",
    "short_liquidations_usd": "",
    "open_price": "",
    "high_price": "",
    "low_price": "",
    "close_price": "",
    "volume": "",
    "open_interest": ""
}

