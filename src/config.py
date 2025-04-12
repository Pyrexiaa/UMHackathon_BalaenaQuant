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

