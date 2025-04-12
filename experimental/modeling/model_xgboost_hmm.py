import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn import hmm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEE_RATE = 0.0006

class XGBHMMEnsemble:
    def __init__(self):
        self.xgb_model = None
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'delta_estimated_leverage_ratio',
            'whale_ratio_diff',
            'close_7d_ma',
            'volume_zscore',
            'net_address_flow'
        ]
        self._validate_paths()

    def _validate_paths(self):
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_PATH = self.PROJECT_ROOT / "experimental/datasets/btc_data_with_target.csv"
        self.MODEL_DIR = self.PROJECT_ROOT / "experimental/modeling/models/xgboost_hmm"
        
        if not self.DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found at {self.DATA_PATH}")
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def prepare_features(self, df):
        """Feature engineering for both training and backtesting"""
        df = df.copy()
        
        # Create required features
        df['close_7d_ma'] = df['close'].rolling(7).mean()
        
        # Volume z-score
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()
        
        # Whale ratio difference
        if 'exchange_whale_ratio' in df.columns and 'start_time_exchange_whale_ratio' in df.columns:
            df['whale_ratio_diff'] = df['exchange_whale_ratio'] - df['start_time_exchange_whale_ratio']
        
        # Leverage ratio difference
        if 'estimated_leverage_ratio' in df.columns and 'start_time_estimated_leverage_ratio' in df.columns:
            df['delta_estimated_leverage_ratio'] = df['estimated_leverage_ratio'] - df['start_time_estimated_leverage_ratio']
        
        # Address flow
        if 'addresses_count_inflow' in df.columns and 'addresses_count_outflow' in df.columns:
            df['net_address_flow'] = df['addresses_count_inflow'] - df['addresses_count_outflow']
        
        # Drop rows with missing values from feature creation
        df = df.dropna(subset=self.feature_names)
        return df

    def _hmm_features(self, X):
        try:
            hmm_features = self.hmm_model.predict_proba(X)
            return hmm_features
        except AttributeError:
            logging.error("HMM model not trained yet")
            raise
        except Exception as e:
            logging.error("HMM feature generation failed", exc_info=True)
            raise

    def train_val_test_split(self, df):
        """Split the processed dataframe by time periods"""
        df = df.sort_index()
        train_data = df['2020-01-01':'2022-12-31']
        validation_data = df['2023-01-01':'2023-12-31']
        test_data = df['2024-01-01':'2025-03-31']
        return train_data, validation_data, test_data

    def train(self, processed_df, n_states=4):
        """Train the ensemble model"""
        try:
            # Process features and handle NaNs
            processed_df = self.prepare_features(processed_df)
            processed_df.index = pd.to_datetime(processed_df.index)

            # Split data
            train_df, val_df, _ = self.train_val_test_split(processed_df)
            
            if len(train_df) == 0 or len(val_df) == 0:
                raise ValueError("Invalid data split - check date ranges")

            # Prepare features and targets
            X_train = train_df[self.feature_names]
            X_val = val_df[self.feature_names]
            y_train = train_df['target']
            y_val = val_df['target']

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train HMM
            logging.info("Training HMM...")
            self.hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")
            self.hmm_model.fit(X_train_scaled)

            # Get HMM features
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
            joblib.dump(self.xgb_model, self.MODEL_DIR / 'xgb_model.pkl')
            joblib.dump(self.hmm_model, self.MODEL_DIR / 'hmm_model.pkl')
            joblib.dump(self.scaler, self.MODEL_DIR / 'scaler.pkl')

            logging.info("Training complete. Models saved.")
            return self

        except Exception as e:
            logging.error("Training failed", exc_info=True)
            raise

    def predict(self, X):
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
            # Prepare features for backtesting
            df = self.prepare_features(raw_df.copy())
            df = df[['close'] + self.feature_names + ['target']].copy()

            # Generate predictions
            X = df[self.feature_names]
            probs = self.predict(X)

            # Create signals dataframe
            signals_df = pd.DataFrame(index=df.index)
            signals_df['price'] = df['close']
            signals_df['buy_prob'] = probs[:, 2]
            signals_df['sell_prob'] = probs[:, 0]
            signals_df['signal'] = 1  # Default to hold

            # Trading parameters
            buy_thresh = 0.3
            sell_thresh = 0.3
            position_size = 0.1

            # Trading simulation
            capital = initial_capital
            position = 0
            equity = []
            trades = []

            spread_threshold = 0.1  # Minimum difference between buy/sell probs

            for i in range(len(signals_df)):
                buy_prob = signals_df.iloc[i]['buy_prob']
                sell_prob = signals_df.iloc[i]['sell_prob']
                
                if (buy_prob > buy_thresh) and (buy_prob - sell_prob > spread_threshold):
                    new_signal = 2  # Buy
                elif (sell_prob > sell_thresh) and (sell_prob - buy_prob > spread_threshold):
                    new_signal = 0  # Sell
                else:
                    new_signal = 1  # Hold

                signals_df.iat[i, signals_df.columns.get_loc('signal')] = new_signal

                # Execute trades
                if i > 0 and new_signal != signals_df.iloc[i-1]['signal']:
                    if position != 0:
                        # Close position
                        capital += position * current_price * (1 - fee)
                        trades.append({
                            'date': signals_df.index[i],
                            'type': 'sell' if position > 0 else 'buy',
                            'price': current_price,
                            'shares': abs(position)
                        })
                        position = 0

                    if new_signal == 2 and capital > 0:
                        # Open new long position
                        position = (capital * position_size) / current_price
                        capital -= position * current_price * (1 + fee)
                        trades.append({
                            'date': signals_df.index[i],
                            'type': 'buy',
                            'price': current_price,
                            'shares': position
                        })

                equity.append(capital + position * current_price)

            # Performance metrics
            returns = pd.Series(equity).pct_change().dropna()
            sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std() if len(returns) > 0 else 0
            max_dd = self._calculate_max_drawdown(equity)

            # Calculate win rate
            win_rate = 0.0
            if trades:
                profitable_trades = [t for t in trades if t.get('profit', 0) > 0]
                win_rate = len(profitable_trades) / len(trades)

            print(f"\nPerformance Summary:")
            print(f"Initial Equity: ${initial_capital:,.2f}")
            print(f"Final Equity: ${equity[-1]:,.2f}" if equity else "No trades executed")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Max Drawdown: {max_dd:.2%}")
            print(f"Total Trades: {len(trades)}")
            print(f"Win Rate: {win_rate:.2%}")

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
            logging.error("Backtest failed", exc_info=True)
            raise

    def _calculate_max_drawdown(self, equity):
        if not equity:
            return 0
        running_max = equity[0]
        drawdowns = []
        for val in equity:
            running_max = max(running_max, val)
            drawdowns.append((running_max - val) / running_max)
        return max(drawdowns)


def main():
    logging.info("Starting model training and backtesting...")
    
    # Initialize model
    ensemble_model = XGBHMMEnsemble()
    
    try:
        # Load and prepare data
        df = pd.read_csv(ensemble_model.DATA_PATH, parse_dates=['datetime'])
        df = df.set_index('datetime').sort_index()
        df = ensemble_model.prepare_features(df)
        
        # Train model
        ensemble_model.train(df)
        
        # Backtest
        logging.info("Running backtest...")
        backtest_results = ensemble_model.backtest(df)
        
        # Display results
        logging.info(f"Backtest completed. Final Equity: ${backtest_results['final_equity']:,.2f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results['signals_df'].index, backtest_results['signals_df']['price'], label='Price')
        plt.plot(backtest_results['signals_df'].index, backtest_results['signals_df']['signal'], label='Signal', alpha=0.5)
        plt.title('Price and Trading Signals')
        plt.legend()
        plt.show()
        
    except Exception as e:
        logging.error("Main execution failed", exc_info=True)
        raise

if __name__ == "__main__":
    main()