import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data."""
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]