import torch
import pandas as pd
import numpy as np

from experimental.modeling.constants import ASSUMPTION_9
from .general_config import (
    MODEL_METRICS_PATH,
    MODEL_OUTPUT_FILE_PATH,
    EVALUATE_PATH,
    FEE_RATE,
    SELL_SIGNAL,
    BUY_SIGNAL,
    HOLD_SIGNAL,
    WINDOW_SIZE
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt

import os
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

# --- Load CSV ---
def load_csv(df_path):
    df = pd.read_csv(df_path)

    # Keep and parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year

    # Drop nan rows
    df = df.dropna()

    # Split by year
    df_train = df[df["year"] < 2024].copy()
    df_test = df[df["year"] >= 2024].copy()

    df_train = df_train.drop(columns=["datetime", "year"], axis=1)
    df_test = df_test.drop(columns=["datetime", "year"], axis=1)

    return df_train, df_test


# --- Normalize ---
def normalize_data(df, previous_scaling=None, scaling_path="scaling.pkl"):
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
        dump(scaler, scaling_path, compress=True)

    # Combine scaled features and label
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df["target"] = target

    return scaled_df

# --- Preprocess data ---
def preprocess_data(df):
    X = []
    y = []
    for i in range(WINDOW_SIZE, len(df)):
        X.append(df.iloc[i - WINDOW_SIZE : i].drop(columns=["target"]).values)
        y.append(df["target"].iloc[i])

    return np.array(X), np.array(y)

def prepare_features(df):
    df = df[ASSUMPTION_9].copy()
    df = df.dropna()  # Drop rows with NaN values
    df = df.reset_index(drop=True)  # Reset index after dropping rows

    return df

def calculate_class_distribution(y_train):
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    class_counts = torch.bincount(
        y_train_tensor
    )  # Counts of each class in the training labels
    total_samples = len(y_train_tensor)
    class_freq = (
        class_counts.float() / total_samples
    )  # Normalize counts by total samples

    # Calculate class weights (inverse of class frequency)
    class_weights = 1.0 / class_freq  # Inverse of frequency
    class_weights = class_weights / class_weights.sum()  # Normalize so they sum to 1

    # Convert to a tensor and ensure it's of float type
    class_weights = class_weights.float()
    return class_weights


def predict_signals_from_probs(probs, buy_thresh=0.30, sell_thresh=0.30):
    """
    Convert class probabilities into trading signals:
    - class 0 (sell): prob > sell_thresh → 0.30
    - class 1 (hold): prob > hold_thresh → 0.40
    - class 2 (buy): prob > buy_thresh → 0.30
    """
    signals = []
    for p in probs:
        p = np.array(p)
        # Apply threshold rules
        if p[0] > sell_thresh and p[2] <= buy_thresh:
            signals.append(SELL_SIGNAL)  # Buy
        elif p[2] > buy_thresh and p[0] <= sell_thresh:
            signals.append(BUY_SIGNAL)  # Sell
        else:
            signals.append(HOLD_SIGNAL)  # Hold
    return np.array(signals)


def backtest(df, signals):
    capital = 1_000_000
    position = 0
    entry_price = None
    equity = []
    trades = []
    trade_dates = []

    # Fixed risk parameters (removed volatility dependency)
    stop_loss = 0.02  # Fixed 2% stop loss
    take_profit = 0.03  # Fixed 3% take profit
    position_size_pct = 0.05  # Fixed 5% position size

    for i in range(1, len(df)):
        price = df.iloc[i]["close"]

        # Exit conditions
        if position != 0:
            pnl = position * (price - entry_price)
            returns = pnl / abs(position * entry_price)

            # Simplified exit logic
            if (
                (returns < -stop_loss)
                or (returns > take_profit)
                or (signals[i] != (2 if position > 0 else 0))
            ):
                # Apply fees and close position
                pnl -= abs(position * price) * FEE_RATE
                capital += pnl
                trades.append(pnl)
                trade_dates.append(df.index[i])
                position = 0

        # Entry conditions - simplified
        if position == 0 and signals[i] != 1:  # Just check if not hold signal
            entry_price = price
            position_size = int((capital * position_size_pct) // price)
            position = position_size if signals[i] == 2 else -position_size
            capital -= abs(position * price) * (1 + FEE_RATE)
            trade_dates.append(df.index[i])

        equity.append(capital + position * price)

    return np.array(equity), trades, trade_dates


def evaluate_performance(equity, trades, set_name="Validation"):
    """Calculate performance metrics"""
    returns = np.diff(equity) / equity[:-1]
    sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
    max_dd = max(
        (peak - val) / peak for peak, val in zip(np.maximum.accumulate(equity), equity)
    )
    win_rate = np.mean(np.array(trades) > 0) if trades else 0

    print(f"\n{set_name} Performance:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2%}")

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": len(trades),
        "win_rate": win_rate,
    }


# --- Evaluation ---
def evaluate_model(model, test_loader, output_dir):
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

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=["Buy", "Hold", "Sell"], output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"Overall Test Accuracy: {acc:.4f}")
    print(
        "Overall Classification Report:\n",
        classification_report(y_true, y_pred, target_names=["Buy", "Hold", "Sell"]),
    )
    print("Overall Confusion Matrix:\n", cm)

    # Convert to numpy and save probabilities and predictions
    df_probs = pd.DataFrame(probs_array, columns=["Prob_Buy", "Prob_Hold", "Prob_Sell"])
    df_probs["True_Label"] = y_true
    df_probs["Predicted_Label"] = y_pred
    df_probs.to_csv(os.path.join(output_dir, MODEL_OUTPUT_FILE_PATH), index=False)
    print(
        f"Saved overall model probabilities and predictions to: {os.path.join(output_dir, MODEL_OUTPUT_FILE_PATH)}"
    )

    # Prepare metrics for saving
    metrics["accuracy"] = acc
    for label, scores in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            metrics[f"precision_{label}"] = scores["precision"]
            metrics[f"recall_{label}"] = scores["recall"]
            metrics[f"f1-score_{label}"] = scores["f1-score"]
            metrics[f"support_{label}"] = scores["support"]
        elif label in ["macro avg", "weighted avg"]:
            metrics[f"precision_{label.replace(' ', '_')}"] = scores["precision"]
            metrics[f"recall_{label.replace(' ', '_')}"] = scores["recall"]
            metrics[f"f1-score_{label.replace(' ', '_')}"] = scores["f1-score"]
            metrics[f"support_{label.replace(' ', '_')}"] = scores["support"]

    cm_df = pd.DataFrame(
        cm,
        index=["Buy", "Hold", "Sell"],
        columns=["Predicted_Buy", "Predicted_Hold", "Predicted_Sell"],
    )

    # Save metrics and confusion matrix to a single CSV file
    metrics_df = pd.DataFrame([metrics])
    with pd.ExcelWriter(os.path.join(output_dir, MODEL_METRICS_PATH)) as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")

    print(
        f"Saved overall model metrics and confusion matrix to: {os.path.join(output_dir, MODEL_METRICS_PATH)}"
    )

    return probs_array, y_true

def evaluate_gnn(model, test_loader, output_dir):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            num_nodes = inputs.shape[2]
            adj = torch.eye(num_nodes).unsqueeze(0).repeat(inputs.size(0), 1, 1)
            logits = model(inputs, adj)
            probs = torch.softmax(logits, dim=1)  # Shape: [batch_size, num_classes]
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_array = np.array(all_probs)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=["Buy", "Hold", "Sell"], output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"Overall Test Accuracy: {acc:.4f}")
    print(
        "Overall Classification Report:\n",
        classification_report(y_true, y_pred, target_names=["Buy", "Hold", "Sell"]),
    )
    print("Overall Confusion Matrix:\n", cm)

    # Convert to numpy and save probabilities and predictions
    df_probs = pd.DataFrame(probs_array, columns=["Prob_Buy", "Prob_Hold", "Prob_Sell"])
    df_probs["True_Label"] = y_true
    df_probs["Predicted_Label"] = y_pred
    df_probs.to_csv(os.path.join(output_dir, MODEL_OUTPUT_FILE_PATH), index=False)
    print(
        f"Saved overall model probabilities and predictions to: {os.path.join(output_dir, MODEL_OUTPUT_FILE_PATH)}"
    )

    # Prepare metrics for saving
    metrics["accuracy"] = acc
    for label, scores in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            metrics[f"precision_{label}"] = scores["precision"]
            metrics[f"recall_{label}"] = scores["recall"]
            metrics[f"f1-score_{label}"] = scores["f1-score"]
            metrics[f"support_{label}"] = scores["support"]
        elif label in ["macro avg", "weighted avg"]:
            metrics[f"precision_{label.replace(' ', '_')}"] = scores["precision"]
            metrics[f"recall_{label.replace(' ', '_')}"] = scores["recall"]
            metrics[f"f1-score_{label.replace(' ', '_')}"] = scores["f1-score"]
            metrics[f"support_{label.replace(' ', '_')}"] = scores["support"]

    cm_df = pd.DataFrame(
        cm,
        index=["Buy", "Hold", "Sell"],
        columns=["Predicted_Buy", "Predicted_Hold", "Predicted_Sell"],
    )

    # Save metrics and confusion matrix to a single CSV file
    metrics_df = pd.DataFrame([metrics])
    with pd.ExcelWriter(os.path.join(output_dir, MODEL_METRICS_PATH)) as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")

    print(
        f"Saved overall model metrics and confusion matrix to: {os.path.join(output_dir, MODEL_METRICS_PATH)}"
    )

    return probs_array, y_true

def plot_results(df, signals, output_dir):
    """Visualize backtest results"""
    plt.figure(figsize=(14, 6))
    plt.plot(df["close"], label="Price", alpha=0.6)

    buy_signals = df.iloc[np.where(signals == 2)]
    sell_signals = df.iloc[np.where(signals == 0)]

    plt.scatter(
        buy_signals.index,
        buy_signals["close"],
        marker="^",
        color="green",
        label="Buy Signal",
        s=50,
    )
    plt.scatter(
        sell_signals.index,
        sell_signals["close"],
        marker="v",
        color="red",
        label="Sell Signal",
        s=50,
    )

    plt.title("Buy and Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, EVALUATE_PATH))
    plt.close()

def interpret_cohens_d(d):
    """Interprets Cohen's d effect size."""
    if abs(d) < 0.2:
        return "(insignificant)"
    elif abs(d) < 0.5:
        return "(small)"
    elif abs(d) < 0.8:
        return "(moderate)"
    else:
        return "(large)"
