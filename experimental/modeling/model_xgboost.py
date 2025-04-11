"""
XGBoost Trading Strategy Implementation
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import json

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data_with_target.csv"
RESULTS_DIR = PROJECT_ROOT / "experimental/modeling/results/xgboost"
MODEL_DIR = PROJECT_ROOT / "experimental/modeling/models"
FEE_RATE = 0.0006

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.class_weights = {0: 5, 1: 1, 2: 5}

    def load_and_preprocess_data(self):
        df = pd.read_csv(DATA_PATH, parse_dates=['datetime']).set_index('datetime').sort_index()

        delta_cols = [c for c in df.columns if c.startswith('start_time_')]
        for col in delta_cols:
            base_col = col.replace('start_time_', '')
            if base_col in df.columns:
                df[f'delta_{base_col}'] = df[base_col] - df[col]

        df['close_7d_ma'] = df['close'].rolling(7).mean()
        df['close_30d_std'] = df['close'].rolling(30).std()
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()

        if 'target' not in df.columns:
            fee = FEE_RATE
            df['future_return'] = df['close'].pct_change().shift(-1)
            df['target'] = 1
            df.loc[df['future_return'] > fee, 'target'] = 2
            df.loc[df['future_return'] < -fee, 'target'] = 0

        if 'target' not in df.columns:
            raise ValueError("Target column could not be found or created!")

        print("\nTarget value counts:")
        print(df['target'].value_counts())

        return df.dropna()

    def train_val_test_split(self, df):
        return (
            df['2020-01-01':'2022-12-31'],
            df['2023-01-01':'2023-12-31'],
            df['2024-01-01':'2025-03-31']
        )

    def prepare_features(self, train, val, test):
        # Add moving stats
        for df in [train, val, test]:
            df['close_7d_ma'] = df['close'].rolling(7).mean()
            df['close_30d_std'] = df['close'].rolling(30).std()
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()

        # Feature engineering: deltas
        for df in [train, val, test]:
            if 'exchange_whale_ratio' in df.columns and 'start_time_exchange_whale_ratio' in df.columns:
                df['whale_ratio_diff'] = df['exchange_whale_ratio'] - df['start_time_exchange_whale_ratio']
            if 'estimated_leverage_ratio' in df.columns and 'start_time_estimated_leverage_ratio' in df.columns:
                df['delta_estimated_leverage_ratio'] = df['estimated_leverage_ratio'] - df['start_time_estimated_leverage_ratio']

            # Price-based features
            df['price_range'] = df['high'] - df['low']
            df['price_momentum'] = df['close'] - df['open']
            df['price_return_ratio'] = df['close'] / df['open']

            # Flow & transfer features
            df['net_address_flow'] = df['addresses_count_inflow'] - df['addresses_count_outflow']
            df['net_transaction_flow'] = df['transactions_count_inflow'] - df['transactions_count_outflow']
            df['transfer_skew'] = df['tokens_transferred_mean'] / df['tokens_transferred_median']
            df['net_liquidations_usd'] = df['long_liquidations_usd'] - df['short_liquidations_usd']

        # Final feature list
        potential_features = [
            'delta_estimated_leverage_ratio', 'whale_ratio_diff', 'close_7d_ma',
            'close_30d_std', 'volume_zscore', 'taker_buy_ratio', 'open_interest',
            'price_range', 'price_momentum', 'price_return_ratio',
            'net_address_flow', 'net_transaction_flow', 'transfer_skew',
            'net_liquidations_usd', 'fees_transaction_mean_usd'
        ]

        self.features = [f for f in potential_features if all(f in df.columns for df in [train, val, test])]
        if not self.features:
            raise ValueError("No common features found across all datasets!")

        print(f"Using features: {self.features}")

        X_train = self.scaler.fit_transform(train[self.features])
        X_val = self.scaler.transform(val[self.features])
        X_test = self.scaler.transform(test[self.features])

        return X_train, X_val, X_test, train['target'], val['target'], test['target']


    def train_model(self, X_train, y_train, X_val, y_val):
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            scale_pos_weight=self.class_weights,
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            eval_metric='mlogloss'
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=20)

    import numpy as np

    def predict_signals(self, X, buy_thresh=0.3, sell_thresh=0.3, hold_thresh=0.4):
        """
        Predicts trading signals based on model probabilities.
        Arguments:
            X: Features for prediction.
            buy_thresh: Probability threshold for 'Buy'.
            sell_thresh: Probability threshold for 'Sell'.
            hold_thresh: Probability threshold for 'Hold'.
        
        Returns:
            signals: List of predicted signals as 0 (Sell), 1 (Hold), or 2 (Buy).
        """
        # Make probability predictions using the trained model
        probs = self.model.predict_proba(X)
        
        # Initialize signals as 1 (Hold) by default
        signals = np.ones(len(probs), dtype=int)  # 1 represents 'Hold'

        # For Buy (class 2), check if the probability surpasses the buy_thresh
        signals[probs[:, 2] > buy_thresh] = 2  # 2 represents 'Buy'

        # For Sell (class 0), check if the probability surpasses the sell_thresh
        signals[probs[:, 0] > sell_thresh] = 0  # 0 represents 'Sell'

        # Ensure that the signals are based on the highest probability class for each row
        signal_class = probs.argmax(axis=1)
        
        for i in range(len(signals)):
            if probs[i, signal_class[i]] >= hold_thresh:
                signals[i] = signal_class[i]  # Keep 0 for Sell, 1 for Hold, 2 for Buy
        
        return signals


    def backtest(self, df, signals):
        capital, position, prev_signal = 1_000_000, 0, 0
        equity, trades = [], []

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

            equity.append(capital + position * price)

        return np.array(equity), trades

    def evaluate_performance(self, equity, trades, set_name="Validation"):
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

    def save_model(self):
        joblib.dump(self.model, MODEL_DIR / "xgboost_model.pkl")
        joblib.dump(self.scaler, MODEL_DIR / "scaler.pkl")
        with open(MODEL_DIR / "features.json", 'w') as f:
            json.dump(self.features, f)

    def plot_results(self, df, equity, signals, set_name="validation"):
        plt.figure(figsize=(14, 7))

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df.index, df['close'], label='Price', alpha=0.5)
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(df.index, equity, label='Equity', color='green')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend(loc='upper right')
        ax1.set_title(f'{set_name.capitalize()} Backtest Results')

        plt.subplot(2, 1, 2)
        plt.step(df.index, signals, where='post', label='Signals')
        plt.yticks([-1, 0, 1], ['Sell', 'Hold', 'Buy'])
        plt.ylabel('Trading Signal')
        plt.xlabel('Date')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{set_name}_results.png")
        plt.close()

def main():
    model = XGBoostTradingModel()
    df = model.load_and_preprocess_data()
    train, val, test = model.train_val_test_split(df)
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_features(train, val, test)
    model.train_model(X_train, y_train, X_val, y_val)

    y_pred = model.predict_signals(X_val)
    val_equity, val_trades = model.backtest(val, y_pred)

    model.evaluate_performance(val_equity, val_trades, set_name="Validation")
    model.plot_results(val, val_equity, y_pred, set_name="Validation")


    model.save_model()

if __name__ == "__main__":
    main()
