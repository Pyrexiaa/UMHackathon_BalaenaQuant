import torch.nn as nn

# --- CNN Model ---
# class CryptoCNN(nn.Module):
#     def __init__(self, input_features, num_classes=3):
#         super(CryptoCNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3)
#         self.relu = nn.ReLU()
#         self.pool = nn.AdaptiveMaxPool1d(1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(64, num_classes)

#     def forward(self, x):
#         # x shape: [batch, time, features] → permute to [batch, features, time]
#         x = x.permute(0, 2, 1)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x
    
# import torch
# import torch.nn as nn

class CryptoCNN(nn.Module):
    def __init__(self, input_features, num_classes=1):
        super(CryptoCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [batch, time, features] → permute to [batch, features, time]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x shape: [batch, features, time] → permute back to [batch, time, features]
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Take the output of the last time step
        x = self.fc(x)
        return x
