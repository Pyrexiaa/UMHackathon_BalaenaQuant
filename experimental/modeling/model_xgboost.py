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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report
import joblib
from joblib import dump, load
import json

import sys

# Add the path to your 'quantpilot' folder dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../quantpilot')))

# from features.technical_indicators import MovingAverageFeature, VolatilityFeature

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data_with_target_technical_hmm_kmeans.csv"
RESULTS_DIR = PROJECT_ROOT / "experimental/modeling/results/xgboost"
MODEL_DIR = Path("quantpilot/models_weights/xgboost")
FEE_RATE = 0.0006
SCALING_PATH = "quantpilot/models_weights/xgboost/scaler.pkl"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.class_weights = {0: 5, 1: 1, 2: 5}

    def load_csv(self,df_path):
        df = pd.read_csv(df_path)
        
        # Keep and parse datetime
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["year"] = df["datetime"].dt.year
        
        # Drop nan rows
        # df = df.dropna()

        # Split by year
        df_train = df[df["year"].between(2020, 2022)].copy()
        df_val = df[df["year"] == 2023].copy()
        df_test = df[df["year"] >= 2024].copy()

        df_train = df_train.drop(columns=["datetime", "year"], axis=1)
        df_val = df_val.drop(columns=["datetime", "year"], axis=1)
        df_test = df_test.drop(columns=["datetime", "year"], axis=1)

        return df_train, df_val, df_test

    def train_val_test_split(self, df):
        return (
            df['2020-01-01':'2022-12-31'],
            df['2023-01-01':'2023-12-31'],
            df['2024-01-01':'2025-03-31']
        )

    def prepare_features(self,df):
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
    
    def normalize_data(self,df, previous_scaling=None):
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
    

    def get_param_grid(self):
        return {
            'learning_rate': [0.01, 0.02],  
            'max_depth': [4, 5, 6],            
            'n_estimators': [50, 100],      
            'subsample': [0.7, 0.6],        
            'colsample_bytree': [0.7, 0.6],
            'min_child_weight': [3],
            'gamma': [0.1],
            'reg_alpha': [0.1],
            'reg_lambda': [0.5]
        }
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        # Create time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Initialize model with fixed params
        base_model = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            tree_method='hist',
            eval_metric=['mlogloss'],
            early_stopping_rounds=50,
            random_state=42
        )
        
        # Setup GridSearch
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.get_param_grid(),
            scoring='neg_log_loss',  # For probabilistic classification
            cv=tscv,
            verbose=3,
            n_jobs=-1,
            refit=True
        )
        
        # Sample weights for class imbalance
        sample_weights = np.where(
            y_train == 0, 1.5, 
            np.where(y_train == 1, 1.0, 1.3)
        )
        
        # Fit with validation data as early stopping
        grid_search.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=10
        )
        
        # Save best model
        self.model = grid_search.best_estimator_
        
        print("\nBest parameters found:")
        print(grid_search.best_params_)
        print(f"Best validation score: {grid_search.best_score_:.4f}")
        
        return grid_search

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


    
    def analyze_clusters(self, df, predictions, cluster_column='kmeans_cluster'):
        """
        Analyzes model predictions by different clusters.
        
        Args:
            df (pd.DataFrame): The original dataset with a cluster column.
            predictions (np.array): Model predictions aligned with df.
            cluster_column (str): The name of the clustering column in df.
        """
        df = df.copy()
        df['predicted_signal'] = predictions

        if cluster_column not in df.columns:
            print(f"Cluster column '{cluster_column}' not found in dataframe.")
            return

        cluster_groups = df.groupby(cluster_column)

        for cluster, group in cluster_groups:
            print(f"\nCluster {cluster} - Sample Size: {len(group)}")
            print("Predicted Signal Distribution:")
            print(group['predicted_signal'].value_counts())
            print("True Target Distribution:")
            print(group['target'].value_counts())

            print("\nClassification Report:")
            print(classification_report(group['target'], group['predicted_signal']))
            

    def backtest(self, df, signals):
        capital = 1_000_000
        position = 0
        entry_price = None
        equity = []
        trades = []
        trade_dates = [] 
        
        for i in range(1, len(df)):  # Avoid lookahead
            price = df.iloc[i]['close']
            volatility = df.iloc[i-1]['volatility_24']  # Use past volatility
            
            # Dynamic risk management
            current_stop_loss = 0.02 if volatility > 0.025 else 0.03
            current_take_profit = 0.03 if volatility > 0.025 else 0.04
            
            # Exit conditions
            if position != 0:
                pnl = position * (price - entry_price)
                returns = pnl / abs(position * entry_price)
                
                if (returns < -current_stop_loss) or \
                (returns > current_take_profit) or \
                (signals[i] != (2 if position > 0 else 0)):
                    
                    pnl -= abs(position * price) * FEE_RATE
                    capital += pnl
                    trades.append(pnl)
                    trade_dates.append(df.index[i])  # Record trade date
                    position = 0
            
            # Entry conditions - only trade strong signals
            if position == 0 and abs(signals[i] - 1) > 0.5:  # Filter weak signals
                entry_price = price
                position_size = (capital * 0.05) // price  # Smaller position size
                position = position_size if signals[i] > 1 else -position_size
                capital -= abs(position * price) * (1 + FEE_RATE)
                trade_dates.append(df.index[i])  # Record entry date
            
            equity.append(capital + position * price)
        
        return np.array(equity), trades, trade_dates

    def evaluate_performance(self, df, equity, trades, trade_dates, set_name="Validation"):

        # print(f"\n--- {set_name} Trade Dates ---")
        # for date in trade_dates:
        #     print(date)
        # print(f"Total trades: {len(trade_dates)}\n")

        # Calculate trading days
        total_days = (df.index[-1] - df.index[0]).days
        unique_trade_days = len(set([d.date() for d in trade_dates]))  # Count unique trading days
        trade_frequency = (unique_trade_days / total_days) * 100  # % of days with trades

        returns = np.diff(equity) / equity[:-1]
        
        # Sharpe Ratio (annualized)
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
        
        # Win Rate
        win_rate = np.mean(np.array(trades) > 0) if trades else 0
        
        # Rest of your metrics...
        print(f"Trade Frequency: {trade_frequency:.2f}% of days had trades")
        
        print(f"\n{set_name} Performance:")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Trade Frequency: {trade_frequency:.2f}% of days")
        
        # Strategy Validation
        passed = (sharpe >= 1.8) and (max_dd <= 0.4) and (trade_frequency >= 3.0)
        print(f"\nStrategy Passed: {'YES' if passed else 'NO'}")
        
        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'trade_frequency': trade_frequency,
            'passed': passed
        }

    def train_model(self, X_train, y_train, X_val, y_val):
        # Ensure DataFrames
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=self.features)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val, columns=self.features)
        
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'learning_rate': 0.01,
            'max_depth': 4,
            'n_estimators': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'min_child_weight': 3,
            'gamma': 0.1,
            'eval_metric': ['mlogloss', 'merror'],
            'early_stopping_rounds': 50
        }
        
        # Class weights
        class_counts = np.bincount(y_train)
        weights = len(y_train) / (3 * class_counts)
        sample_weights = np.array([weights[label] for label in y_train])
        
        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=10
        )

    def save_model(self):
        joblib.dump(self.model, MODEL_DIR / "xgboost_model.pkl")
        joblib.dump(self.scaler, MODEL_DIR /"scaler.pkl") 
        # joblib.dump(self.scaler, MODEL_DIR / "scaler.pkl")
        # with open(MODEL_DIR / "features.json", 'w') as f:
        #     json.dump(self.features, f)

    def evaluate_performance(self,equity, trades, set_name="Validation"):
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
    dataset_path = "experimental/datasets/btc_data_with_target_technical_hmm_kmeans.csv"
    model = XGBoostTradingModel()
    train , val , test = model.load_csv(dataset_path)
    # train, val, test = model.train_val_test_split(df)
    
    # Prepare features (now returns DataFrames)
    train_feat = model.prepare_features(train)
    val_feat = model.prepare_features(val)
    test_feat = model.prepare_features(test)

    # Set the features list for scaling
    # model.features = train_feat.columns.tolist()

    # Normalize data
    train_feat = model.normalize_data(train_feat)
    val_feat = model.normalize_data(val_feat, SCALING_PATH)
    test_feat = model.normalize_data(test_feat, SCALING_PATH)

    
    # Run tuning
    # grid_search = model.tune_hyperparameters(X_train.values, y_train, X_val.values, y_val)
    
    # Train with proper feature names
    model.features = train_feat.columns.tolist()
    X_train = train_feat
    y_train = train_feat["target"]
    X_val = val_feat
    y_val = val_feat["target"]
    X_test = test_feat
    y_test = test_feat["target"]
    model.train_model(X_train.values, y_train.values, X_val.values, y_val.values)

    # Validation set
    signals = model.predict_signals(X_val.values)
    equity, trades, trade_dates = model.backtest(val, signals)
    val_results = model.evaluate_performance(equity, trades,"Validation")
    
    # Test set
    signals = model.predict_signals(X_test.values)
    equity, trades, trade_dates = model.backtest(test, signals)
    test_results = model.evaluate_performance( equity, trades,"Test")
    
    # Save model
    model.save_model()

     # Evaluate performance
    model.evaluate_performance(equity, trades,"Test")

if __name__ == "__main__":
    main()
