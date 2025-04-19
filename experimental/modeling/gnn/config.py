FEE_RATE = 0.0006
SELL_SIGNAL = 0
BUY_SIGNAL = 2
HOLD_SIGNAL = 1

SEQUENCE_LENGTH = 4
PREDICTION_LENGTH = 4
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 1
WEIGHT_DECAY = 1e-3
NUM_LAYERS = 2
DROPOUT_RATE = 0.5
HIDDEN_FEATURES = 32
RAYTUNE_SAMPLES = 5

# --- Paths ---
SCALING_PATH = "output/gnn/scaling_values.pkl"
MODEL_OUTPUT_PATH = "output/gnn/model_output.pth"
MODEL_OUTPUT_FILE_PATH = "output/gnn/model_output.csv"
TRADING_OUTPUT_FILE_PATH = "output/gnn/trading_output.csv"
EVALUATE_PATH = "output/gnn/evaluate_graph.png"