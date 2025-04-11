import torch.nn as nn

# --- Hyperparameters ---
WINDOW_SIZE = 24  # number of time steps used for prediction
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

# --- Model Configuration ---
LOSS_FUNCTION = nn.MSELoss()

# --- Paths ---
SCALING_PATH = "output/cnn/scaling_values.pkl"
OUTPUT_PATH = "output/cnn/model_output.csv"