import torch
import torch.nn as nn

FEE_RATE = 0.0006
SELL_SIGNAL = 2
BUY_SIGNAL = 0
HOLD_SIGNAL = 1

# --- Hyperparameters ---
WINDOW_SIZE = 4  # number of time steps used for prediction
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

# --- Model Configuration ---
class_weights = torch.tensor([1.0, 2.0, 2.0]).float()
LOSS_FUNCTION = nn.CrossEntropyLoss(weight=class_weights)

# --- Paths ---
SCALING_PATH = "output/cnn/scaling_values.pkl"
MODEL_OUTPUT_PATH = "output/cnn/model_output.pth"
MODEL_OUTPUT_FILE_PATH = "output/cnn/model_output.csv"
TRADING_OUTPUT_FILE_PATH = "output/cnn/trading_output.csv"
EVALUATE_PATH = "output/cnn/evaluate_graph.png"

SELECTED_FEATURES = ["datetime", "target", "close", "open", "high", "low", "volume", "future_return", "open_interest"]