class BaseConfig:
    FEE_RATE = 0.0006
    BUY_SIGNAL = 0
    HOLD_SIGNAL = 1
    SELL_SIGNAL = 2
    BUY_THRESHOLD = 0.30
    SELL_THRESHOLD = 0.30
    WINDOW_SIZE = 4


class TCNConfig:
    TCN_MODEL_PATH = "src/models_weights/tcn/model.pth"
    TCN_SCALER_PATH = "src/models_weights/tcn/scaler.pkl"


class XGBConfig:
    XGB_MODEL_PATH = "src/models_weights/xgboost/xgboost_model.pkl"
    XGB_SCALER_PATH = "src/models_weights/xgboost/scaler.pkl"
    XGB_FEATURE_PATH = [
        "price_change_1",
        "ema_5_8_13_cross",
        "taker_sell_ratio",
        "taker_buy_ratio",
        "taker_buy_sell_ratio",
        "rsi_14",
        "rsi_obv_signal_14",
        "bb_signal_20",
        "coinbase_premium_index_usdt_adjusted",
        "macd_signal_flag",
        "coinbase_premium_gap_usdt_adjusted",
        "macd_trade_signal",
        "macd",
        "addresses_count_sender",
        "addresses_count_active",
        "blockreward",
        "tokens_transferred_mean",
        "long_liquidations",
        "addresses_count_receiver",
        "taker_sell_volume"
    ]


class CNNConfig:
    CNN_MODEL_PATH = "src/models_weights/cnn/model.pth"
    CNN_SCALER_PATH = "src/models_weights/cnn/scaler.pkl"


class GNNConfig:
    HIDDEN_FEATURES = 4
    GNN_MODEL_PATH = "src/models_weights/gnn/model.pth"
    GNN_SCALER_PATH = "src/models_weights/gnn/scaler.pkl"


class TabNetConfig:
    TABNET_MODEL_PATH = "src/models_weights/tabnet/model.zip"
    TABNET_SCALER_PATH = "src/models_weights/tabnet/scaler.pkl"
    TABNET_FEATURE_NAMES_PATH = "src/models_weights/xgboost/features.json"
