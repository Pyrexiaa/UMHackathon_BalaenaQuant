import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(FeatureEncoder, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, encoded_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        return self.proj(self.norm(x))

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                    stride=stride, padding=0, dilation=dilation))
        self.layernorm1 = nn.LayerNorm(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=0, dilation=dilation))
        self.layernorm2 = nn.LayerNorm(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # Manually pad for 'same' output length
        pad = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (pad, 0))  # left padding only (causal)

        out = self.conv1(x_padded)
        out = out.transpose(1, 2)
        out = self.layernorm1(out)
        out = out.transpose(1, 2)
        out = self.dropout1(out)

        out = F.pad(out, (pad, 0))  # same padding again before 2nd conv
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.layernorm2(out)
        out = out.transpose(1, 2)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.3):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size-1)*dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CryptoSignalModel(nn.Module):
    def __init__(self, num_classes=3, encoder_dim=32, tcn_channels=[128, 64], tcn_dropout=0.3, head_dropout=0.4, head_hidden_dim=32):
        super(CryptoSignalModel, self).__init__()

        # Feature encoders per group
        self.enc_buyer_strength = FeatureEncoder(2, encoder_dim)
        self.enc_institution = FeatureEncoder(2, encoder_dim)
        self.enc_supply = FeatureEncoder(2, encoder_dim)
        self.enc_retail = FeatureEncoder(3, encoder_dim)
        self.enc_market = FeatureEncoder(5, encoder_dim)

        total_encoded_dim = encoder_dim * 5

        # TCN backbone
        self.tcn = TCN(input_size=total_encoded_dim, output_size=tcn_channels[-1],
                       num_channels=tcn_channels, kernel_size=3, dropout=tcn_dropout)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(tcn_channels[-1], head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_classes)
        )

    def forward(self, x):
        buyer = self.enc_buyer_strength(torch.cat([x['exchange_whale_ratio'], x['taker_buy_ratio']], dim=-1))
        inst = self.enc_institution(torch.cat([x['coinbase_premium_gap'], x['coinbase_premium_index']], dim=-1))
        supply = self.enc_supply(torch.cat([x['exchange_supply_ratio'], x['miner_supply_ratio']], dim=-1))
        retail = self.enc_retail(torch.cat([x['addresses_count_active'], x['addresses_count_outflow'], x['transactions_count_outflow']], dim=-1))
        market = self.enc_market(torch.cat([x['tokens_transferred_total'], x['short_liquidations'], x['short_liquidations_usd'],
                                            x['long_liquidations'], x['long_liquidations_usd']], dim=-1))

        x_concat = torch.cat([buyer, inst, supply, retail, market], dim=-1)
        x_tcn = x_concat.permute(0, 2, 1)
        out = self.tcn(x_tcn)
        return self.head(out)

