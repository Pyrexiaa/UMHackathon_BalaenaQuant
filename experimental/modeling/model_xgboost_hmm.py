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
        self.feature_names = [
            'delta_leverage',
            'delta_whale_ratio',
            'close_7d_ma',
            'volume_zscore',
            'net_address_flow'
        ]
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
        processed_df.index = pd.to_datetime(processed_df.index)

        try:
            # Split based on time, not random
            train_df = processed_df['2020-01-01':'2022-12-31']
            val_df = processed_df['2023-01-01':'2023-12-31']

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
        try:
            # Process data and validate
            df = self._safe_feature_engineering(raw_df.copy())
            df = df[['close'] + self.feature_names + ['target']].copy()
            
            # Generate predictions
            X = df[self.feature_names]
            probs = self.predict(X)
            
            # Create signals DataFrame
            signals_df = pd.DataFrame(index=df.index)
            signals_df['price'] = df['close']
            signals_df['buy_prob'] = probs[:, 2]
            signals_df['sell_prob'] = probs[:, 0]
            signals_df['signal'] = 1  # Default hold

            print(signals_df.head(10))
            
            # Strategy parameters
            buy_thresh = 0.35  # Tune these
            sell_thresh = 0.35
            position_size = 0.1  # Risk management
            
            # Backtest variables
            capital = initial_capital
            position = 0
            equity = []
            trades = []
            
            for i in range(len(signals_df)):
                current_price = signals_df.iloc[i]['price']
                
                # Generate signal
                if signals_df.iloc[i]['buy_prob'] > buy_thresh:
                    new_signal = 2  # Buy
                elif signals_df.iloc[i]['sell_prob'] > sell_thresh:
                    new_signal = 0  # Sell
                else:
                    new_signal = 1  # Hold
                    
                signals_df.iloc[i, signals_df.columns.get_loc('signal')] = new_signal
                
                # Execute trades
                if new_signal != signals_df.iloc[i-1]['signal'] if i > 0 else True:
                    # Close previous position
                    if position != 0:
                        capital += position * current_price * (1 - fee)
                        trades.append({
                            'date': signals_df.index[i],
                            'type': 'sell' if position > 0 else 'buy',
                            'price': current_price,
                            'shares': abs(position)
                        })
                        position = 0
                    
                    # Open new position
                    if new_signal == 2 and capital > 0:  # Buy
                        position = (capital * position_size) / current_price
                        capital -= position * current_price * (1 + fee)
                        trades.append({
                            'date': signals_df.index[i],
                            'type': 'buy',
                            'price': current_price,
                            'shares': position
                        })
                    elif new_signal == 0 and capital > 0:  # Sell only if holding
                        pass  # Removed short selling for safety
                        
                # Track equity
                equity.append(capital + position * current_price)
            
            # Calculate performance metrics
            returns = pd.Series(equity).pct_change().dropna()
            sharpe = np.sqrt(252*24) * returns.mean() / returns.std() if len(returns) > 0 else 0
            max_dd = self._calculate_max_drawdown(equity)
            
            # Calculate win rate
            win_rate = 0.0
            if len(trades) > 0:
                profitable_trades = [t for t in trades if t.get('profit', 0) > 0]
                win_rate = len(profitable_trades)/len(trades)
            
            # Print metrics
            print(f"Performance:")
            print(f"Initial Equity: ${initial_capital:,.2f}")
            print(f"Final Equity: ${equity[-1]:,.2f}" if equity else "No trades executed")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Max Drawdown: {max_dd:.2%}")
            print(f"Total Trades: {len(trades)}")
            print(f"Win Rate: {win_rate:.2%}")
            
            # Add profit/loss to trades
            for i in range(1, len(trades)):
                if trades[i]['type'] == 'sell':
                    prev_buy = next((t for t in trades[:i][::-1] if t['type'] == 'buy'), None)
                    if prev_buy:
                        trades[i]['profit'] = (trades[i]['price'] - prev_buy['price']) * trades[i]['shares']
            
            return {
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'initial_equity': initial_capital,
                'final_equity': equity[-1] if equity else initial_capital,
                'trades': trades,
                'signals_df': signals_df
            }
        
        except Exception as e:
                logging.error("Backtest data processing failed", exc_info=True)
                raise

    def _calculate_max_drawdown(self, equity):
        
        """Safe drawdown calculation"""
        if not equity:
            return 0
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd


    def _plot_results(self, signals_df, equity):
        """Enhanced visualization"""
        plt.figure(figsize=(14, 7))
        
        # Price and signals
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(signals_df['price'], label='Price')
        ax1.scatter(signals_df[signals_df['signal'] == 2].index, 
                    signals_df[signals_df['signal'] == 2]['price'], 
                    color='green', label='Buy', marker='^')
        ax1.scatter(signals_df[signals_df['signal'] == 0].index,
                    signals_df[signals_df['signal'] == 0]['price'],
                    color='red', label='Sell', marker='v')
        ax1.set_title('Price with Trading Signals')
        ax1.legend()
        
        # Equity curve
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(equity, label='Portfolio Value')
        ax2.set_title('Equity Curve')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.MODEL_DIR / 'backtest_results.png')
        

def main():
    # Instantiate the model
    model = XGBHMMEnsemble()

    # Load your raw data (must match expected structure)
    df = pd.read_csv(model.DATA_PATH, index_col=0)
    
    # Feature engineering
    processed_df = model._safe_feature_engineering(df)

    # Train the model
    model.train(processed_df)

    # Optional: Backtest and evaluate
    results = model.backtest(df)
    
    # Plot equity curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(results['signals_df'].index, results['signals_df']['price'], label='BTC Price')
    plt.plot(results['signals_df'].index, results['signals_df']['signal'], label='Signal')
    plt.title("Signals Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
