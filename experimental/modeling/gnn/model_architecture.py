import torch
import torch.nn as nn

# --- Graph Neural Network Architecture for Time Series ---
class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [batch_size, num_nodes, in_features]
        # adj: [batch_size, num_nodes, num_nodes]
        support = torch.matmul(adj, x)
        output = self.linear(support)
        return output

class TimeSeriesGNN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_features, out_features, num_layers=2, sequence_length=4):
        super(TimeSeriesGNN, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.layers = nn.ModuleList([GNNLayer(in_features if i == 0 else hidden_features, hidden_features) for i in range(num_layers)])
        self.fc_out = nn.Linear(hidden_features * num_nodes * sequence_length, out_features)

    def forward(self, x, adj):
        # x: [batch_size, seq_len, num_nodes] -> Reshape to [batch_size, seq_len, num_nodes, 1]
        batch_size, seq_len, num_nodes = x.shape
        x = x.unsqueeze(-1) # Add a dimension of size 1 for in_features

        # Now x has shape [(batch_size, seq_len), num_nodes, 1]
        x = x.reshape(batch_size * seq_len, num_nodes, self.in_features)
        adj = adj.repeat(seq_len, 1, 1) # Repeat adjacency matrix for each time step

        for layer in self.layers:
            x = torch.relu(layer(x, adj))

        # Reshape back and prepare for output layer
        x = x.reshape(batch_size, seq_len, num_nodes, self.hidden_features)
        # Aggregate across time and nodes
        x = x.reshape(batch_size, -1) # [batch_size, seq_len * num_nodes * hidden_features]

        return self.fc_out(x)

