from quantpilot.data import DataLoader

loader = DataLoader()
metrics = [
    "price-ohlcv", "taker-buy-sell-stats", "addresses_count_inflow", "addresses_count",
    "coinbase_premium_index", "estimated_leverage_ratio", "exchange_whale_ratio",
    "tokens_transferred", "blockreward", "fees_transaction", "miner_supply_ratio", 
    "exchange_supply_ratio", "transactions_count_inflow", "liquidations", "open_interest", "difficulty"
]
df = loader.run(metrics)