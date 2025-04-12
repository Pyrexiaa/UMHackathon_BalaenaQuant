class Config:
    FEE_RATE = 0.0006
    BUY_SIGNAL = 0
    HOLD_SIGNAL = 1
    SELL_SIGNAL = 2
    BUY_THRESHOLD = 0.30
    SELL_THRESHOLD = 0.30
    WINDOW_SIZE = 4
    TCN_MODEL_PATH = "src/models/trained/tcn/model.pth"
    TCN_SCALER_PATH = "src/models/trained/tcn/scaler.pkl"

    XGB_MODEL_PATH = "src/models/trained/xgboost/model.pkl"
    XGB_SCALER_PATH = "src/models/trained/xgboost/scaler.pkl"
    XGB_FEATURE_PATH = "src\models\xgb_features.json"

    TCN_FEATURES_PATH = "src/models/trained/tcn/features.json"
    XGB_CLASS_WEIGHTS = {0: 1, 1: 1, 2: 1}
    TCN_CLASS_WEIGHTS = {0: 1, 1: 1, 2: 1}
    XGB_MODEL_PARAMS = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss'
    }
    TCN_MODEL_PARAMS = {
        'num_channels': [64, 128, 64],
        'kernel_size': 3,
        'dropout': 0.2,
        'num_classes': 3,
        'input_features': 12
    }
    TCN_CLASS_WEIGHTS = {0: 1, 1: 1, 2: 1}
    XGB_CLASS_WEIGHTS = {0: 1, 1: 1, 2: 1}
    XGB_N_SPLITS = 3
    TCN_N_SPLITS = 3
    XGB_BUY_THRESH = 0.3
    XGB_SELL_THRESH = 0.3
    TCN_BUY_THRESH = 0.3
    TCN_SELL_THRESH = 0.3
    XGB_VERBOSE = 10
    TCN_VERBOSE = 10
    XGB_EVAL_SET = None
    TCN_EVAL_SET = None
    XGB_DEVICE = None
    TCN_DEVICE = None


