class BaseConfig:
    FEE_RATE = 0.0006
    THRESHOLD = 0.375
    WINDOW_SIZE = 120


class TCNConfig:
    TCN_MODEL_PATH = "quantpilot/models_weights/tcn/checkpoint.pt"
    TCN_SCALER_PATH = "quantpilot/models_weights/tcn/scaling_values.pkl"
    # TCN_SCALER_PATH = "quantpilot/models_weights/tcn/scaling_values.pkl"
    # TCN_MODEL_PATH = "quantpilot/models_weights/tcn/model.pth"
    

class TabNetConfig:
    TABNET_MODEL_PATH = "quantpilot/models_weights/tabnet/tabnet_fold1.zip"
    TABNET_SCALER_PATH = "quantpilot/models_weights/tabnet/scaler.pkl"
    

class XGBConfig:
    XGB_MODEL_PATH = "quantpilot/models_weights/xgboost_final_1/xgboost_model.pkl"
    XGB_SCALER_PATH = "quantpilot/models_weights/xgboost_final_1/scaler.pkl"

class CNNConfig:
    CNN_MODEL_PATH = "quantpilot/models_weights/cnn/model.pth"
    CNN_SCALER_PATH = "quantpilot/models_weights/cnn/scaler.pkl"


class GNNConfig:
    HIDDEN_FEATURES = 4
    GNN_MODEL_PATH = "quantpilot/models_weights/gnn/checkpoint.pt"
    GNN_SCALER_PATH = "quantpilot/models_weights/gnn/scaling_values.pkl"

