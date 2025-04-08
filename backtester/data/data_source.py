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
