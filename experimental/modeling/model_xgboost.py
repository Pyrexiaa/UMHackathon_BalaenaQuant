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
import joblib  # For model saving
import json

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Goes up 3 levels from experimental/modeling/

DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data.csv"
RESULTS_DIR = PROJECT_ROOT / "experimental/modeling/results/xgboost"
MODEL_DIR = PROJECT_ROOT / "experimental/modeling/models"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Better than os.makedirs for Path objects
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEE_RATE = 0.0006  # 0.06% trading fee

class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.class_weights = {0: 5, 1: 1, 2: 5}   # Sell, Hold, Buy

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
        df = df.set_index('datetime').sort_index()

        # Print available columns for debugging
        print("Available columns in dataset:")
        print(df.columns.tolist())
    
    # 1. Create delta features for existing columns
        delta_cols = [c for c in df.columns if c.startswith('start_time_')]
        for col in delta_cols:
            base_col = col.replace('start_time_', '')
            if base_col in df.columns:
                df[f'delta_{base_col}'] = df[base_col] - df[col]

    # 2. Create temporal features
        df['close_7d_ma'] = df['close'].rolling(7).mean()
        df['close_30d_std'] = df['close'].rolling(30).std()
        df['volume_zscore'] = ((df['volume'] - df['volume'].rolling(30).mean()) / 
                          df['volume'].rolling(30).std())
    
    # 3. Create target variable (MOST IMPORTANT FIX)
        fee = 0.0006  # 0.06% trading fee
    
    # Calculate future returns (next period's return)
        df['future_return'] = df['close'].pct_change().shift(-1)
    
    # Create ternary labels (-1: sell, 0: hold, 1: buy)
        df['target'] = 1  # Default to hold
        df.loc[df['future_return'] > fee, 'target'] = 2   # Buy signal
        df.loc[df['future_return'] < -fee, 'target'] = 0  # Sell signal
    
    # Verify target column was created
        if 'target' not in df.columns:
            raise ValueError("Failed to create target column!")
    
        print("\nTarget value counts:")
        print(df['target'].value_counts())
    
        return df.dropna()

    def train_val_test_split(self, df):
        """Split data into train/validation/test sets"""
        train = df['2020-01-01':'2022-12-31']
        val = df['2023-01-01':'2023-12-31']
        test = df['2024-01-01':'2025-03-31']
        return train, val, test

    def prepare_features(self, train, val, test):

        for df in [train, val, test]:
            if 'close_7d_ma' not in df.columns:
                df['close_7d_ma'] = df['close'].rolling(7).mean()
            if 'close_30d_std' not in df.columns:
                df['close_30d_std'] = df['close'].rolling(30).std()
            if 'volume_zscore' not in df.columns:
                df['volume_zscore'] = ((df['volume'] - df['volume'].rolling(30).mean()) / 
                                  df['volume'].rolling(30).std())
   
        if ('exchange_whale_ratio' in df.columns and 
            'start_time_exchange_whale_ratio' in df.columns):
            df['whale_ratio_diff'] = (df['exchange_whale_ratio'] - 
                                     df['start_time_exchange_whale_ratio'])
        
        # Calculate delta_estimated_leverage_ratio if components exist
        if ('estimated_leverage_ratio' in df.columns and 
            'start_time_estimated_leverage_ratio' in df.columns):
            df['delta_estimated_leverage_ratio'] = (df['estimated_leverage_ratio'] - 
                                                  df['start_time_estimated_leverage_ratio'])


        common_features = []
        potential_features = [
        'delta_estimated_leverage_ratio',
        'whale_ratio_diff',
        'close_7d_ma',
        'close_30d_std',
        'volume_zscore',
        'taker_buy_ratio',
        'open_interest'
    ]

        for feature in potential_features:
            if all(feature in df.columns for df in [train, val, test]):
                common_features.append(feature)
    
        if not common_features:
            raise ValueError("No common features found across all datasets!")
    
        print(f"Using features: {common_features}")
        self.features = common_features

        # Normalization
        X_train = self.scaler.fit_transform(train[self.features])
        X_val = self.scaler.transform(val[self.features])
        X_test = self.scaler.transform(test[self.features])

        y_train = train['target']
        y_val = val['target']
        y_test = test['target']

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, X_train, y_train, X_val, y_val):
        self.model = XGBClassifier(  # Use the correctly imported class
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
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=20
        )

    def predict_signals(self, X, buy_thresh=0.15, sell_thresh=0.15):
        """Generate trading signals from probabilities"""
        probs = self.model.predict_proba(X)
        signals = []
        for p in probs:
            if p[2] > buy_thresh:    # Buy (class 1)
                signals.append(1)
            elif p[0] > sell_thresh: # Sell (class -1)
                signals.append(-1)
            else:
                signals.append(0)
        return np.array(signals)

    def backtest(self, df, signals):
        """Backtest trading strategy"""
        capital = 1_000_000  # Starting capital
        position = 0
        prev_signal = 0
        equity = []
        trades = []
        
        for i in range(len(df)):
            price = df.iloc[i]['close']
            
            # Close existing position if signal changes
            if prev_signal != 0 and signals[i] != prev_signal:
                pnl = position * (price - entry_price) - abs(position * price) * FEE_RATE
                capital += pnl
                position = 0
                trades.append(pnl)
            
            # Open new position
            if signals[i] != 0 and position == 0:
                entry_price = price
                position = (capital // price) * signals[i]  # Long/Short
                capital -= abs(position * price) * FEE_RATE
                prev_signal = signals[i]
            
            equity.append(capital + position * price)
        
        return np.array(equity), trades

    def evaluate_performance(self, equity, trades, set_name="Validation"):
        """Calculate performance metrics"""
        returns = np.diff(equity) / equity[:-1]
        
        # Sharpe Ratio
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
        
        # Max Drawdown
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        # Trade stats
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
        """Save model and artifacts"""
        joblib.dump(self.model, f"{MODEL_DIR}/xgboost_model.pkl")
        joblib.dump(self.scaler, f"{MODEL_DIR}/scaler.pkl")
        
        # Save feature list
        with open(f"{MODEL_DIR}/features.json", 'w') as f:
            json.dump(self.features, f)

    def plot_results(self, df, equity, signals, set_name="validation"):
        """Visualize backtest results"""
        plt.figure(figsize=(14, 7))
        
        # Price and Equity
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df.index, df['close'], label='Price', alpha=0.5)
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
        plt.yticks([-1, 0, 1], ['Sell', 'Hold', 'Buy'])
        plt.ylabel('Trading Signal')
        plt.xlabel('Date')
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{set_name}_results.png")
        plt.close()

def main():
    # Initialize and run pipeline
    model = XGBoostTradingModel()
    
    try:
        df = model.load_and_preprocess_data()
        train, val, test = model.train_val_test_split(df)
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_features(train, val, test)
        model.train_model(X_train, y_train, X_val, y_val)
    
    # Validation
        val_signals = model.predict_signals(X_val)
        val_equity, val_trades = model.backtest(val, val_signals)
        model.evaluate_performance(val_equity, val_trades, "Validation")
        model.plot_results(val, val_equity, val_signals, "validation")
    
    # Test
        test_signals = model.predict_signals(X_test)
        test_equity, test_trades = model.backtest(test, test_signals)
        model.evaluate_performance(test_equity, test_trades, "Test")
        model.plot_results(test, test_equity, test_signals, "test")
    
    # Save model
        model.save_model()
        print("\nModel training and evaluation complete!")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()