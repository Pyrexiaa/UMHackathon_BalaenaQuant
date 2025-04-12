FEE_RATE = 0.0006
SELL_SIGNAL = 2
BUY_SIGNAL = 0
HOLD_SIGNAL = 1

SEQUENCE_LENGTH = 4
PREDICTION_LENGTH = 4
BATCH_SIZE = 64
LR = 0.0001
EPOCHS = 50
WEIGHT_DECAY = 1e-5
MODEL_DIMENSION = 64
NUM_LAYERS = 2
DROPOUT_RATE = 0.5
HIDDEN_FEATURES = 6

# --- Paths ---
SCALING_PATH = "output/timesnet/scaling_values.pkl"
MODEL_OUTPUT_PATH = "output/timesnet/model_output.pth"
MODEL_OUTPUT_FILE_PATH = "output/timesnet/model_output.csv"
TRADING_OUTPUT_FILE_PATH = "output/timesnet/trading_output.csv"
EVALUATE_PATH = "output/timesnet/evaluate_graph.png"