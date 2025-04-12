import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    WINDOW_SIZE,
    SCALING_PATH,
    MODEL_OUTPUT_PATH,
    MODEL_OUTPUT_FILE_PATH,
    TRADING_OUTPUT_FILE_PATH,
    EVALUATE_PATH,
    FEE_RATE,
    SELL_SIGNAL,
    BUY_SIGNAL,
    HOLD_SIGNAL,
)
from .model_architecture import TCNClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from joblib import dump, load
from .datasets import TimeSeriesDataset
from torch.utils.data import DataLoader

def prepare_features(df):
    potential_features = [
        'future_return', 'price_change_1', 'ema_5_8_13_cross', 'taker_sell_ratio', 
        'taker_buy_ratio', 'taker_buy_sell_ratio', 'rsi_14', 'rsi_obv_signal_14', 
        'bb_signal_20', 'coinbase_premium_index_usdt_adjusted', 'macd_signal_flag', 
        'coinbase_premium_gap_usdt_adjusted', 'macd_trade_signal', 'macd', 
        'addresses_count_sender', 'addresses_count_active', 'blockreward', 
        'tokens_transferred_mean', 'long_liquidations', 'addresses_count_receiver', "target"
    ]

    df = df[potential_features].copy()
    df = df.dropna()  # Drop rows with NaN values
    df = df.reset_index(drop=True)  # Reset index after dropping rows

    return df

def calculate_class_distribution(y_train):
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    class_counts = torch.bincount(y_train_tensor)  # Counts of each class in the training labels
    total_samples = len(y_train_tensor)
    class_freq = class_counts.float() / total_samples  # Normalize counts by total samples

    # Calculate class weights (inverse of class frequency)
    class_weights = 1.0 / class_freq  # Inverse of frequency
    class_weights = class_weights / class_weights.sum()  # Normalize so they sum to 1

    # Convert to a tensor and ensure it's of float type
    class_weights = class_weights.float()
    return class_weights

# --- Load CSV ---
def load_csv(df_path):
    df = pd.read_csv(df_path)
    
    # Keep and parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    
    # Drop nan rows
    df = df.dropna()

    # Split by year
    df_train = df[df["year"].between(2020, 2022)].copy()
    df_val = df[df["year"] == 2023].copy()
    df_test = df[df["year"] >= 2024].copy()

    df_train = df_train.drop(columns=["datetime", "year"], axis=1)
    df_val = df_val.drop(columns=["datetime", "year"], axis=1)
    df_test = df_test.drop(columns=["datetime", "year"], axis=1)

    return df_train, df_val, df_test

# --- Normalize ---
def normalize_data(df, previous_scaling=None):
    # Separate features and label
    features = df.drop(columns=["target"])
    target = df["target"].reset_index(drop=True)

    if previous_scaling:
        # Load previously saved scaler
        scaler = load(previous_scaling)
        scaled_features = scaler.transform(features)
    else:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        dump(scaler, SCALING_PATH, compress=True)

    # Combine scaled features and label
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df["target"] = target

    return scaled_df

# --- Preprocess data ---
def preprocess_data(df):
    X = []
    y = []
    for i in range(WINDOW_SIZE, len(df)):
        X.append(df.iloc[i - WINDOW_SIZE:i].drop(columns=["target"]).values)
        y.append(df["target"].iloc[i])
    
    return np.array(X), np.array(y)

# --- Convert to Torch tensors ---
def convert_to_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Wrap into datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, X_test, y_test

# --- Training Loop ---
def train_model(model, train_loader, val_loader, optimizer, loss_criterion):
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation loop
        # TODO: Add early stopping etc
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_preds = model(val_x)
                val_loss += loss_criterion(val_preds, val_y).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}")

    return model

def predict_signals_from_probs(probs, buy_thresh=0.30, sell_thresh=0.30):
    """
    Convert class probabilities into trading signals:
    - class 0 (buy): prob > buy_thresh → 0.30
    - class 2 (sell): prob > sell_thresh → 0.30
    - class 1 (hold): prob > hold_thresh → 0.40
    """
    signals = []
    for p in probs:
        p = np.array(p)
        if p[0] > buy_thresh:
            signals.append(BUY_SIGNAL)  # Buy
        elif p[2] > sell_thresh:
            signals.append(SELL_SIGNAL)  # Sell
        else:
            signals.append(HOLD_SIGNAL)  # Hold
    return np.array(signals)

def backtest(df, signals):
    """Backtest strategy based on trading signals (0: buy, 1: hold, 2: sell)."""
    capital = 1000000
    position = 0
    entry_price = 0
    prev_signal = 1  # Initialize with hold signal
    equity = []
    trades = []
    pnl_list = []
    pos_list = []
    capital_list = []
    trades_flag = []

    for i in range(len(signals)):
        price = df.iloc[i]["close"]
        pnl = 0
        trade_occurred = 0

        # Exit condition
        if position != 0 and signals[i] != prev_signal:
            if position > 0:  # Long position
                pnl = position * (price - entry_price) - abs(position * price) * FEE_RATE
            elif position < 0:  # Short position
                pnl = position * (entry_price - price) - abs(position * price) * FEE_RATE
            capital += pnl
            trades.append(pnl)
            position = 0  # Close position
            trade_occurred = 1  # Trade happened

        # Entry/Adding to position condition
        if signals[i] == 0:  # Buy signal
            available_capital = capital - abs(position) * price * FEE_RATE if position != 0 else capital
            if available_capital > 0:
                additional_shares = available_capital // price
                if additional_shares > 0:
                    capital -= additional_shares * price * FEE_RATE
                    if position >= 0:
                        new_total_shares = abs(position) + additional_shares
                        entry_price = (abs(position) * entry_price + additional_shares * price) / new_total_shares if abs(position) > 0 else price
                        position = new_total_shares
                    else:  # Closing short and opening long
                        pnl = position * (entry_price - price) - abs(position * price) * FEE_RATE
                        capital += pnl
                        trades.append(pnl)
                        position = additional_shares
                        entry_price = price
                        trade_occurred = 1
            prev_signal = signals[i]
        elif signals[i] == 2:  # Sell signal
            available_capital = capital - abs(position) * price * FEE_RATE if position != 0 else capital
            if available_capital > 0:
                additional_shares = available_capital // price
                if additional_shares > 0:
                    capital -= additional_shares * price * FEE_RATE
                    if position <= 0:
                        new_total_shares = abs(position) + additional_shares
                        entry_price = (abs(position) * entry_price + additional_shares * price) / new_total_shares if abs(position) > 0 else price
                        position = -new_total_shares
                    else:  # Closing long and opening short
                        pnl = position * (price - entry_price) - abs(position * price) * FEE_RATE
                        capital += pnl
                        trades.append(pnl)
                        position = -additional_shares
                        entry_price = price
                        trade_occurred = 1
            prev_signal = signals[i]
        elif signals[i] == 1:
            prev_signal = signals[i] # Hold signal

        # Logging per step
        current_equity = capital
        if position > 0:
            current_equity += position * price
        elif position < 0:
            current_equity += position * entry_price - position * price

    # TODO: Edit backtest formula
    capital, position, prev_signal = 1_000_000, 0, 0
    equity, trades, pnl_list, pos_list, capital_list, trades_flag = [], [], [], [], [], []

    for i in range(len(df)):
        price = df.iloc[i]['close']

        if prev_signal != 0 and signals[i] != prev_signal:
            pnl = position * (price - entry_price) - abs(position * price) * FEE_RATE
            capital += pnl
            position = 0
            trades.append(pnl)

        if signals[i] != 0 and position == 0:
            entry_price = price
            position = (capital // price) * signals[i]
            capital -= abs(position * price) * FEE_RATE
            prev_signal = signals[i]

        # Apply Stop-Loss and Take-Profit:
        if position != 0:
            # Check for stop-loss condition (e.g., 3% loss)
            if (price - entry_price) / entry_price < -0.03:
                pnl = position * (price - entry_price) - abs(position * price) * FEE_RATE
                capital += pnl
                position = 0
                trades.append(pnl)
                prev_signal = 0  # Reset signal after stop-loss

            # Check for take-profit condition (e.g., 5% gain)
            elif (price - entry_price) / entry_price > 0.05:
                pnl = position * (price - entry_price) - abs(position * price) * FEE_RATE
                capital += pnl
                position = 0
                trades.append(pnl)
                prev_signal = 0  # Reset signal after take-profit

        equity.append(capital + position * price)
        pnl_list.append(pnl)
        pos_list.append(position)
        capital_list.append(capital)
        trades_flag.append(trade_occurred)

    result_df = df.copy().reset_index(drop=True)
    result_df["signal"] = signals
    result_df["equity"] = equity
    result_df["pnl"] = pnl_list
    result_df["capital"] = capital_list
    result_df["position"] = pos_list
    result_df["trades"] = trades_flag  # 1 if a trade happened, else 0
    result_df.to_csv(TRADING_OUTPUT_FILE_PATH, index=False)

    return np.array(equity), trades



def evaluate_performance(equity, trades, set_name="Validation"):
    """Calculate performance metrics"""
    returns = np.diff(equity) / equity[:-1]
    sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
    max_dd = max((peak - val) / peak for peak, val in zip(np.maximum.accumulate(equity), equity))
    win_rate = np.mean(np.array(trades) > 0) if trades else 0

    print(f"\n{set_name} Performance:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2%}")

    return {
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'num_trades': len(trades),
        'win_rate': win_rate
    }

# --- Evaluation ---
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:

            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)  # Shape: [batch_size, num_classes]
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_array = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Buy", "Hold", "Sell"])
    cm = confusion_matrix(y_true, y_pred)

    print(f"Overall Test Accuracy: {acc:.4f}")
    print("Overall Classification Report:\n", report)
    print("Overall Confusion Matrix:\n", cm)

    # Convert to numpy and save
    df_probs = pd.DataFrame(probs_array, columns=["Prob_Buy", "Prob_Hold", "Prob_Sell"])
    df_probs["True_Label"] = y_true
    df_probs["Predicted_Label"] = y_pred
    df_probs.to_csv(MODEL_OUTPUT_FILE_PATH, index=False)
    print(f"Saved overall model probabilities and predictions to: {MODEL_OUTPUT_FILE_PATH}")

    return probs_array, y_true
    
def plot_results(df, equity, signals, set_name="validation"):
    """Visualize backtest results"""
    plt.figure(figsize=(14, 7))
    
    # Price and Equity
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['close'], label='Price', alpha=0.6)
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(df.index, equity, label='Equity', color='green')
    ax2.set_ylabel('Portfolio Value')
    ax2.legend(loc='upper right')
    ax1.set_title(f'{set_name.capitalize()} Backtest Results')

    # Signals
    plt.subplot(2, 1, 2)
    plt.step(df.index, signals, where='post', label='Signals')
    plt.yticks([0, 1, 2], ['Buy', 'Hold', 'Sell'])
    plt.ylabel('Trading Signal')
    plt.xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(EVALUATE_PATH)
    plt.close()

if __name__ == "__main__":
    dataset_path = "experimental/datasets/btc_data_with_target_technical_hmm_kmeans.csv"
    # --- Load CSV ---
    raw_df_train, raw_df_val, raw_df_test = load_csv(dataset_path)
    # --- Prepare features ---
    df_train = prepare_features(raw_df_train)
    df_val = prepare_features(raw_df_val)
    df_test = prepare_features(raw_df_test)
    # # --- Normalize ---
    scaled_train = normalize_data(df_train, previous_scaling=False)
    scaled_val = normalize_data(df_val, previous_scaling=SCALING_PATH)
    scaled_test = normalize_data(df_test, previous_scaling=SCALING_PATH)
    # # --- Preprocess data ---
    X_train, y_train = preprocess_data(scaled_train)
    X_val, y_val = preprocess_data(scaled_val)
    X_test, y_test = preprocess_data(scaled_test)
    # --- Convert to Torch tensors and DataLoaders ---
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    input_features = X_train.shape[2]
    num_channels = [64, 128, 64]  # Define the number of channels in each TCN block
    model = TCNClassifier(input_features, 3, num_channels)
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

    class_weights = calculate_class_distribution(y_train)
    loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model = train_model(model, train_loader, val_loader, optimizer, loss_criterion)

    # Get predicted probabilities
    probs, y_true = evaluate_model(model, test_loader)

    # Generate trading signals
    signals = predict_signals_from_probs(probs, buy_thresh=0.30, sell_thresh=0.30)

    # Backtest on raw (non-scaled) test set
    df_test_raw = raw_df_test.reset_index(drop=True).iloc[WINDOW_SIZE:].copy()
    equity, trades = backtest(df_test_raw, signals)

    # Evaluate performance
    evaluate_performance(equity, trades, set_name="Test")

    # Plot result
    plot_results(df_test_raw, equity, signals, set_name="Test")
