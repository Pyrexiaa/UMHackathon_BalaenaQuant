FEE_RATE = 0.0006
SELL_SIGNAL = 2
BUY_SIGNAL = 0
HOLD_SIGNAL = 1

WINDOW_SIZE = 4
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 1
WEIGHT_DECAY = 1e-5
MODEL_DIMENSION = 64
NUM_LAYERS = 2
DROPOUT_RATE = 0.5
NUM_KERNELS = 6

# --- Paths ---
SCALING_PATH = "output/timesnet/scaling_values.pkl"
MODEL_OUTPUT_PATH = "output/timesnet/model_output.pth"
MODEL_OUTPUT_FILE_PATH = "output/timesnet/model_output.csv"
TRADING_OUTPUT_FILE_PATH = "output/timesnet/trading_output.csv"
EVALUATE_PATH = "output/timesnet/evaluate_graph.png"

class Configs:
    def __init__(self, seq_len, enc_in, d_model, num_class, e_layers, dropout, top_k, d_ff, num_kernels, embed='fixed', freq='h'):
        self.task_name = 'classification'
        self.seq_len = seq_len
        self.label_len = 0  # Not relevant for classification
        self.pred_len = 0   # Not relevant for classification
        self.enc_in = enc_in  # Number of input features
        self.c_out = num_class # Number of output channels (equal to num_class for classification)
        self.d_model = d_model # Dimension of the model's embeddings
        self.embed = embed     # Type of embedding
        self.freq = freq       # Frequency of the time series
        self.dropout = dropout
        self.e_layers = e_layers # Number of encoder layers (TimesBlocks)
        self.num_class = num_class # Number of classes for classification
        self.top_k = top_k
        self.d_ff = d_ff
        self.num_kernels = num_kernels