"""
XGBoost + HMM Ensemble Trading Model
"""
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn import hmm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class XGBHMMEnsemble:
    def __init__(self):
        self.xgb_model = None
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self._validate_paths()
        
    def _validate_paths(self):
        """Handle path validation with error checking"""
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_PATH = self.PROJECT_ROOT / "experimental/datasets/btc_data.csv"
        self.MODEL_DIR = self.PROJECT_ROOT / "experimental/modeling/models/xgboost_hmm"
        
        if not self.DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found at {self.DATA_PATH}")
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def _safe_feature_engineering(self, df):
        """Robust feature creation with validation"""
        required_columns = {
            'start_time_estimated_leverage_ratio', 'estimated_leverage_ratio',
            'start_time_exchange_whale_ratio', 'exchange_whale_ratio',
            'close', 'volume', 'addresses_count_inflow', 'addresses_count_outflow'
        }
        
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        try:
            df = df.copy()
            # Delta features
            df['delta_leverage'] = df['estimated_leverage_ratio'] - df['start_time_estimated_leverage_ratio']
            df['delta_whale_ratio'] = df['exchange_whale_ratio'] - df['start_time_exchange_whale_ratio']
            
            # Temporal features
            df['close_7d_ma'] = df['close'].rolling(7).mean()
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()
            df['net_address_flow'] = df['addresses_count_inflow'] - df['addresses_count_outflow']
            
            # Target engineering with fee consideration
            fee = 0.0006
            df['future_return'] = df['close'].pct_change().shift(-1)
            df['target'] = np.select(
                [df['future_return'] > fee, df['future_return'] < -fee],
                [2, 0],  # Buy=2, Sell=0
                default=1  # Hold=1
            )
            
            return df.dropna()
        
        except Exception as e:
            logging.error("Feature engineering failed", exc_info=True)
            raise

    def _hmm_features(self, X):
        """Generate HMM regime probabilities with validation"""
        try:
            hmm_features = self.hmm_model.predict_proba(X)
            return hmm_features
        except AttributeError:
            logging.error("HMM model not trained yet")
            raise
        except Exception as e:
            logging.error("HMM feature generation failed", exc_info=True)
            raise

    def train(self, processed_df, val_start='2023-01-01', n_states=4):
        """Train on processed data with time-based split"""
        try:
            # Split based on time, not random
            train_df = processed_df.loc[:val_start]
            val_df = processed_df.loc[val_start:]
            
            if len(train_df) == 0 or len(val_df) == 0:
                raise ValueError("Invalid split - check dates")

            X_train = train_df[self.feature_names]
            X_val = val_df[self.feature_names]
            y_train = train_df['target']
            y_val = val_df['target']
            
            # Scaling
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train HMM
            logging.info("Training HMM...")
            self.hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")
            self.hmm_model.fit(X_train_scaled)
            
            # Generate HMM features
            hmm_train = self._hmm_features(X_train_scaled)
            hmm_val = self._hmm_features(X_val_scaled)
            
            # Combine features
            X_train_combined = np.hstack([X_train_scaled, hmm_train])
            X_val_combined = np.hstack([X_val_scaled, hmm_val])
            
            # Train XGBoost
            logging.info("Training XGBoost...")
            self.xgb_model = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                early_stopping_rounds=50,
                eval_metric='mlogloss'
            )
            
            self.xgb_model.fit(
                X_train_combined, y_train,
                eval_set=[(X_val_combined, y_val)],
                verbose=20
            )
            
            # Save models
            joblib.dump(self.xgb_model, self.MODEL_DIR/'xgb_model.pkl')
            joblib.dump(self.hmm_model, self.MODEL_DIR/'hmm_model.pkl')
            joblib.dump(self.scaler, self.MODEL_DIR/'scaler.pkl')
            logging.info("Training complete. Models saved.")
            
            return self
            
        except Exception as e:
            logging.error("Training failed", exc_info=True)
            raise

    def prepare_test_data(self, raw_df):
        """Process raw test data through feature pipeline"""
        return self._safe_feature_engineering(raw_df.copy())    

    def predict(self, X):
        """Safe prediction with validation"""
        try:
            X_scaled = self.scaler.transform(X)
            hmm_features = self._hmm_features(X_scaled)
            combined = np.hstack([X_scaled, hmm_features])
            return self.xgb_model.predict_proba(combined)
        except Exception as e:
            logging.error("Prediction failed", exc_info=True)
            raise

    def backtest(self, raw_df, initial_capital=1_000_000, fee=0.0006):
        """Robust backtesting with integrated feature engineering"""
        try:
            # 1. Process raw data through feature pipeline
            df = self._safe_feature_engineering(raw_df.copy())
            
            # 2. Validate features exist
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                raise KeyError(f"Missing features in processed data: {missing_features}")
            
            # 3. Generate predictions
            X = df[self.feature_names]
            probs = self.predict(X)
            
            # Strategy logic
            positions = []
            capital = initial_capital
            position = 0
            prev_signal = 0
            
            for i in range(len(df)):
                # Signal generation
                buy_prob = probs[i, 2]
                sell_prob = probs[i, 0]
                
                if buy_prob > 0.3:  # Threshold for buy
                    signal = 2
                elif sell_prob > 0.3:  # Threshold for sell
                    signal = 0
                else:
                    signal = 1
                
                # Execution logic
                price = df.iloc[i]['close']
                
                if signal != prev_signal and prev_signal != 1:
                    # Close existing position
                    capital += position * price * (1 - fee)
                    position = 0
                
                if signal == 2 and position == 0:  # Buy
                    position = capital / price
                    capital = 0
                elif signal == 0 and position == 0:  # Sell
                    position = -capital / price
                    capital = 0
                
                positions.append(capital + position * price)
                prev_signal = signal
            
            # Performance metrics
            returns = pd.Series(positions).pct_change()
            sharpe = np.sqrt(365) * returns.mean() / returns.std()
            max_dd = (positions - np.maximum.accumulate(positions)).min()
            
            logging.info(f"Backtest Results:\n"
                         f"Sharpe Ratio: {sharpe:.2f}\n"
                         f"Max Drawdown: {max_dd:.2%}\n"
                         f"Final Equity: {positions[-1]:.2f}")
            
            # Plotting
            plt.figure(figsize=(12,6))
            plt.plot(df.index, positions)
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(True)
            plt.savefig(self.MODEL_DIR/'backtest_results.png')
            
            return {
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'equity_curve': positions
            }
            
        except Exception as e:
            logging.error("Backtest failed", exc_info=True)
            raise
        

if __name__ == "__main__":
    try:
        model = XGBHMMEnsemble()
        raw_df = pd.read_csv(model.DATA_PATH, parse_dates=['datetime'])
        full_processed = model._safe_feature_engineering(raw_df)
        
        # Time-based split
        model.train(
            processed_df=full_processed,
            val_start='2023-01-01'  # Explicit split point
        )
        
        # Backtest on unseen 2024 data
        test_df = full_processed.loc['2024-01-01':]
        model.backtest(test_df)
            
    except Exception as e:
        logging.error("Main execution failed", exc_info=True)
        raise