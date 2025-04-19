import torch

class GroupedFeatureTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        """
        X: ndarray of shape (samples, seq_len, feature_dim)
        y: ndarray of shape (samples,)
        """
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        sample = self.X[idx]  # shape (seq_len, num_features)

        # Create feature groups
        x_dict = {
            'exchange_whale_ratio': sample[:, [0]],
            'taker_buy_ratio': sample[:, [1]],
            'coinbase_premium_gap': sample[:, [2]],
            'coinbase_premium_index': sample[:, [3]],
            'exchange_supply_ratio': sample[:, [4]],
            'miner_supply_ratio': sample[:, [5]],
            'addresses_count_active': sample[:, [6]],
            'addresses_count_outflow': sample[:, [7]],
            'transactions_count_outflow': sample[:, [8]],
            'tokens_transferred_total': sample[:, [9]],
            'short_liquidations': sample[:, [10]],
            'short_liquidations_usd': sample[:, [11]],
            'long_liquidations': sample[:, [12]],
            'long_liquidations_usd': sample[:, [13]],
        }

        # Convert to torch tensors
        x_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in x_dict.items()}
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_dict, y_tensor

    def __len__(self):
        return len(self.X)
