FEE_RATE = 0.0006
SELL_SIGNAL = 2
BUY_SIGNAL = 0
HOLD_SIGNAL = 1

WINDOW_SIZE = 4
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50
WEIGHT_DECAY = 1e-5

# --- Paths ---
SCALING_PATH = "output/tcn/scaling_values.pkl"
MODEL_OUTPUT_PATH = "output/tcn/model_output.pth"
MODEL_OUTPUT_FILE_PATH = "output/tcn/model_output.csv"
TRADING_OUTPUT_FILE_PATH = "output/tcn/trading_output.csv"
EVALUATE_PATH = "output/tcn/evaluate_graph.png"