ASSUMPTION_1 = ["exchange_whale_ratio", "taker_buy_ratio", "target"]

ASSUMPTION_2 = [
    "difficulty",
    "estimated_leverage_ratio",
    "addresses_count_active",
    "addresses_count_receiver",
    "addresses_count_sender",
    "exchange_supply_ratio",
    "addresses_count_inflow",
    "addresses_count_outflow",
    "transactions_count_inflow",
    "transactions_count_outflow",
    "miner_supply_ratio",
    "blockreward",
    "blockreward_usd",
    "fees_transaction_mean",
    "fees_transaction_median",
    "tokens_transferred_mean",
    "tokens_transferred_median",
    "tokens_transferred_total",
    "target",
]

ASSUMPTION_3 = [
    "addresses_count_active",
    "addresses_count_receiver",
    "addresses_count_sender",
    "addresses_count_inflow",
    "addresses_count_outflow",
    "transactions_count_inflow",
    "transactions_count_outflow",
    "exchange_supply_ratio",
    "target",
]

ASSUMPTION_4 = [
    "coinbase_premium_gap",
    "coinbase_premium_gap_usdt_adjusted",
    "coinbase_premium_index",
    "coinbase_premium_index_usdt_adjusted",
    "target",
]

ASSUMPTION_5 = ["miner_supply_ratio", "target"]

ASSUMPTION_6 = [
    "transactions_count_inflow",
    "transactions_count_outflow",
    "addresses_count_inflow",
    "addresses_count_outflow",
    "exchange_supply_ratio",
    "target",
]

ASSUMPTION_7 = [
    "long_liquidations",
    "short_liquidations",
    "long_liquidations_usd",
    "short_liquidations_usd",
    "target",
]

ASSUMPTION_8 = [
    "tokens_transferred_mean",
    "tokens_transferred_median",
    "tokens_transferred_total",
    "target",
]

ALL_ON_CHAIN_FEATURES = ["difficulty", "estimated_leverage_ratio", "addresses_count_active", "addresses_count_sender", "addresses_count_receiver", "exchange_whale_ratio", "coinbase_premium_gap", "coinbase_premium_index", "coinbase_premium_gap_usdt_adjusted", "coinbase_premium_index_usdt_adjusted", "taker_buy_volume", "taker_sell_volume", "taker_buy_ratio", "taker_sell_ratio", "taker_buy_sell_ratio", "blockreward", "blockreward_usd", "fees_transaction_mean", "fees_transaction_mean_usd", "fees_transaction_median", "fees_transaction_median_usd", "miner_supply_ratio", "addresses_count_inflow", "addresses_count_outflow", "exchange_supply_ratio", "transactions_count_inflow", "tokens_transferred_total", "tokens_transferred_mean", "tokens_transferred_median", "transactions_count_inflow", "transactions_count_outflow", "long_liquidations", "short_liquidations", "long_liquidations_usd", "short_liquidations_usd", "open_price", "high_price", "low_price", "close_price", "volume", "open_interest"]
