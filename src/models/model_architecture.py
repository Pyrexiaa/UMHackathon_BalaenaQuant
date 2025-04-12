import torch.nn as nn

# --- Temporal Convolutional Network (TCN) Module ---
class Chomp1d(nn.Module):
    """Removes padding to ensure output has the same length as input in causal convolutions."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """A block consisting of two temporal convolutional layers with residual connections."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """Temporal Convolutional Network architecture."""
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Input x has shape (batch_size, seq_length, input_size)
        # We need to transpose it to (batch_size, input_size, seq_length) for Conv1D
        return self.network(x.transpose(1, 2))

class TCNClassifier(nn.Module):
    """TCN model for multiclass classification."""
    def __init__(self, input_features, num_classes, num_channels, kernel_size=2, dropout=0.2):
        super(TCNClassifier, self).__init__()
        self.tcn = TCN(input_features, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # Input x has shape (batch_size, seq_length, input_features)
        tcn_output = self.tcn(x)  # Output shape: (batch_size, num_channels[-1], seq_length)
        # We need to take the output of the last time step for classification
        last_output = tcn_output[:, :, -1] # Shape: (batch_size, num_channels[-1])
        return self.linear(last_output)