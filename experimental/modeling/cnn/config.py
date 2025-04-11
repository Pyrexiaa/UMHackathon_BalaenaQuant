import torch.nn as nn

# --- Hyperparameters ---
WINDOW_SIZE = 12  # number of time steps used for prediction
BATCH_SIZE = 32
EPOCHS = 100
LR = 5e-4

# --- Model Configuration ---
LOSS_FUNCTION = nn.CrossEntropyLoss()

# --- Paths ---
SCALING_PATH = "output/cnn/scaling_values.pkl"
OUTPUT_PATH = "output/cnn/model_output.pth"
EVALUATE_PATH = "output/cnn/evaluate_graph.png"