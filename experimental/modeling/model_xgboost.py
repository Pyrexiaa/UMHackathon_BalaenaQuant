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

import sys

# Add the path to your 'src' folder dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# from features.technical_indicators import MovingAverageFeature, VolatilityFeature

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data_with_target_technical_hmm_kmeans.csv"
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

    # def prepare_features(self, train, val, test):
    #     # Add dynamic features to all datasets
    #     for df in [train, val, test]:
    #         df['price_range'] = df['high'] - df['low']
    #         df['price_momentum'] = df['close'] - df['open']

    #     # Updated feature list with technical indicators
    #     potential_features = [
    #         # Price action
    #         'close', 'price_range', 'price_momentum',
            
    #         # Technical indicators (optimized)
    #         'sma_50', 'sma_200', 'ema_5', 'rsi_14', 'macd', 'macd_signal',  # Removed some of the redundant EMAs
    #         'macd_crossover', 'volatility_20', 'price_change_1',
            
    #         # Existing features
    #         'volume_zscore', 'net_liquidations_usd', 'taker_buy_ratio'
    #     ]

    #     # Final feature selection
    #     self.features = [f for f in potential_features 
    #                     if all(f in d.columns for d in [train, val, test])]
        
    #     # Scaling and return
    #     X_train = self.scaler.fit_transform(train[self.features])
    #     X_val = self.scaler.transform(val[self.features])
    #     X_test = self.scaler.transform(test[self.features])
        
    #     return X_train, X_val, X_test, train['target'], val['target'], test['target']

    def prepare_features(self, train, val, test):
        # Use all pre-engineered features from the new dataset
        self.features = [
            # Technical indicators
            'rsi_obv_signal', 'bollinger_upper', 'bollinger_lower', 
            'macd', 'kmeans_cluster', 'hmm_state',
            
            # Sentiment and volume features
            'sentiment_score', 'volume_zscore', 'taker_buy_ratio',
            
            # Price action
            'price_range', 'price_momentum', 'close',
            
            # Advanced features
            'volatility_20', 'volume_obv', 'liquidation_skew'
        ]

        # Verify feature existence
        available_features = []
        for f in self.features:
            if all(f in df.columns for df in [train, val, test]):
                available_features.append(f)
            else:
                print(f"Warning: {f} missing in some datasets")
        
        self.features = available_features
        print(f"Final features: {self.features}")

        # Scale features
        X_train = self.scaler.fit_transform(train[self.features])
        X_val = self.scaler.transform(val[self.features])
        X_test = self.scaler.transform(test[self.features])
        
        return X_train, X_val, X_test, train['target'], val['target'], test['target']

    def train_model(self, X_train, y_train, X_val, y_val):
        self.model =XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            scale_pos_weight=self.class_weights,
            min_child_weight=3,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_estimators=50,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            eval_metric='mlogloss'
        ) 

        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=20)

    def predict_signals(self, X, buy_thresh=0.3, sell_thresh=0.3, hold_thresh=0.4):
        """
        Predicts trading signals based on model probabilities.
        Arguments:
            X: Features for prediction.
            buy_thresh: Probability threshold for 'Buy'.
            sell_thresh: Probability threshold for 'Sell'.
            hold_thresh: Probability threshold for 'Hold'        
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

    def backtest(self, df, signals, stop_loss=0.03, take_profit=0.05):
        capital, position, entry_price, prev_signal = 1_000_000, 0, None , 0
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

            # Apply Stop-Loss and Take-Profit:
            if position != 0:
                # Check for stop-loss condition (e.g., 3% loss)
                if (price - entry_price) / entry_price < -stop_loss:
                    pnl = position * (price - entry_price) - abs(position * price) * FEE_RATE
                    capital += pnl
                    position = 0
                    trades.append(pnl)
                    prev_signal = 0  # Reset signal after stop-loss

                # Check for take-profit condition (e.g., 5% gain)
                elif (price - entry_price) / entry_price > take_profit:
                    pnl = position * (price - entry_price) - abs(position * price) * FEE_RATE
                    capital += pnl
                    position = 0
                    trades.append(pnl)
                    prev_signal = 0  # Reset signal after take-profit

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
        plt.step(df.index, signals, where='post', label='Trading Signals')
        plt.ylabel('Signals')
        plt.xlabel('Date')
        plt.title(f'{set_name.capitalize()} Trading Signals')
        plt.legend()

        plt.tight_layout()
        plt.show()


def plot_signals(df, signals):
    plt.figure(figsize=(14, 6))
    plt.plot(df['close'], label='Price', alpha=0.6)

    buy_signals = df.iloc[np.where(signals == 2)]
    sell_signals = df.iloc[np.where(signals == 0)]

    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=50)
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=50)

    plt.title("Buy and Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    model = XGBoostTradingModel()

    # Load and preprocess data
    df = model.load_and_preprocess_data()

    # Train-test-validation split
    train, val, test = model.train_val_test_split(df)

    # Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_features(train, val, test)

    # Train model
    model.train_model(X_train, y_train, X_val, y_val)

    # Predict trading signals
    signals = model.predict_signals(X_val)

    # Backtest strategy
    equity, trades = model.backtest(val, signals)

    # Evaluate performance
    model.evaluate_performance(equity, trades, set_name="Test")

    # Save the model
    model.save_model()

    plot_signals(val, signals)

    # Plot results
    # model.plot_results(test, equity, signals, set_name="Test")

if __name__ == "__main__":
    main()
