import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, source: str, symbol: str, timeframe: str = '15m'):
        """
        Initialize the DataHandler.

        :param source: Path to the CSV data file (or a data source identifier).
        :param symbol: Trading pair symbol (e.g., BTC/USDT).
        :param timeframe: Data timeframe (default '15m').
        """
        self.source = source
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = pd.DataFrame()

    def load_data(self):
        """
        Load data from a CSV file or other source.
        Expected CSV columns: timestamp, open, high, low, close, volume

        This method converts timestamp column to datetime and sets it as index.
        """
        try:
            # TODO: Change this to call API and get data
            self.data = pd.read_csv(self.source)
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                self.data.set_index('timestamp', inplace=True)
            self.data.sort_index(inplace=True)
            print(f"✅ Data loaded successfully from {self.source}.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def clean_data(self):
        """
        Simple cleaning: fill missing values and remove duplicates.
        """
        self.data = self.data.drop_duplicates()
        self.data = self.data.fillna(method='ffill')
        print("✅ Data cleaned (duplicates removed and missing data forward-filled).")

    def add_features(self):
        """
        Compute technical indicators and features for the model:
          - Returns, log returns
          - Volatility over 24 hours and 7 days (assuming constant frequency)
          - Moving averages and exponential moving averages
          - RSI and MACD (approximations for demo purposes)
        """
        df = self.data.copy()

        # TODO: Can add other features if needed
        # Calculate simple returns and log returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Calculate volatility (standard deviation of returns)
        # Adjust the window size based on your data frequency
        window_24h = 24  # Example: 24 periods representing 24*15m=6h, change based on actual frequency
        window_7d = 7 * 24  # Example: Adapt window size based on real frequency/duration
        df['volatility_24h'] = df['returns'].rolling(window=window_24h).std()
        df['volatility_7d'] = df['returns'].rolling(window=window_7d).std()

        # Simple Moving Averages (SMA)
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Exponential Moving Average (EMA) for a fast indicator, e.g., EMA-20
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

        # Calculate RSI (Relative Strength Index)
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        window = 14
        roll_up = up.rolling(window=window).mean()
        roll_down = down.rolling(window=window).mean()
        rs = roll_up / roll_down
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        # Calculate MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26

        # Save the engineered feature set
        self.data = df.dropna().copy()
        print("Feature engineering completed: returns, volatilities, SMA, EMA, RSI, MACD added.")

    def get_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame.
        """
        return self.data

    def run(self):
        """
        Convenience method to load, clean, and preprocess data.
        """
        self.load_data()
        self.clean_data()
        self.add_features()
        return self.get_data()


# Example usage:
if __name__ == "__main__":
    # Adjust the file path and symbol as required.
    dh = DataHandler(source="data/market_data.csv", symbol="BTC/USDT", timeframe="15m")
    processed_data = dh.run()
    print(processed_data.head())
