import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from hmmlearn import hmm
import tensorflow as tf

# Feature Engineering
df = pd.read_csv("experimental/datasets/btc_data_with_target_modified.csv")
df['6_hour_lag_coinbase'] = df['coinbase_premium_index'].shift(6)
df['24_hour_lag_coinbase'] = df['coinbase_premium_index'].shift(24)
df['taker_buy_sell_ratio_sma_6'] = df['taker_buy_sell_ratio'].rolling(window=6).mean()
df['taker_buy_sell_ratio_sma_24'] = df['taker_buy_sell_ratio'].rolling(window=24).mean()
df['price_std_6hr'] = df['close'].rolling(window=6).std()
df['price_std_24'] = df['close'].rolling(window=24).std()
df = df.iloc[24:]  # Drop first 24 rows to get rid of NaN values
df.to_csv("data.csv", index=False)

# Data Preprocessing
def prepare_trading_data(df, split_date='2024-03-25 00:00:00', 
                        n_future_bars=72,
                        tp_threshold=0.045,
                        sl_threshold=-0.05,
                        volatility_threshold=2.3):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    
    df['future_return'] = df['close'].pct_change(n_future_bars).shift(-n_future_bars)
    rolling_vol = df['price_std_6hr']
    vol_filter = rolling_vol > (rolling_vol.mean() * volatility_threshold)
    
    conditions = [
        (df['future_return'] > tp_threshold) & (~vol_filter),  # Long
        (df['future_return'] < sl_threshold) & (~vol_filter),  # Short
        (vol_filter)      # Neutral
    ]
    
    df['3class_signal'] = np.select(conditions, [1, -1, 0], default=0)
    
    train = df.loc[:split_date]
    test = df.loc[split_date:]
    
    feature_cols = [col for col in df.columns if col not in ['future_return', '3class_signal', 'start_time', 'close', 'datetime']]
    
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_train, y_test = train['3class_signal'], test['3class_signal']
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    train_regime = X_train.copy()
    test_regime = X_test.copy()
    
    valid_idx = X_train_scaled.index.intersection(y_train.dropna().index)
    X_train_scaled = X_train_scaled.loc[valid_idx]
    y_train = y_train.loc[valid_idx]
    
    return X_train_scaled, X_test_scaled, y_train, y_test, train_regime, test_regime, scaler

df = pd.read_csv('data.csv')
X_train, X_test, y_train, y_test, train_regime, test_regime, scaler = prepare_trading_data(df)

# Feature Importance and PCA
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
sorted_idx = importances.argsort()[::-1]

# Feature Importance Plot
plt.figure(figsize=(12, 6))
plt.bar(range(X_train.shape[1]), importances[sorted_idx], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_idx], rotation=90)
plt.title("Feature Importance Scores")
plt.tight_layout()
plt.show()


# Remove rows with NaN values
X_train_clean = X_train.dropna()

# PCA for Dimensionality Reduction
pca = PCA().fit(X_train_clean)
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title("PCA Variance Explained")
plt.grid()
plt.show()


# Remove rows with NaN values
train_regime_clean = train_regime.dropna()

# KMeans for Regime Detection
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(train_regime_clean)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title("Elbow Method for Optimal Cluster Number")
plt.grid()
plt.show()

# Trading Model using HMM and CNN
class SelfLearningTradingModel:
    def __init__(self, n_regimes=3, cnn_lookback=60, feature_cols=None):
        self.hmm = hmm.GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=1000)
        self.cnn = None
        self.scaler = None
        self.feature_cols = feature_cols or [
            'exchange_whale_ratio', 
            'coinbase_premium_index',
            'taker_buy_sell_ratio_sma_6',
            'taker_buy_sell_ratio_sma_24',
            'price_std_24', 
            'netflow', 
            'taker_buy_sell_ratio'
        ]
        self.lookback = cnn_lookback
        self.current_regime = None

    def _build_cnn(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model

    def train(self, X_train_scaled, y_train, train_regime, existing_scaler):

        # Remove rows with NaN values in specific columns
        train_regime_clean = train_regime.dropna(subset=self.feature_cols)

        self.scaler = existing_scaler
        self.hmm.fit(train_regime_clean[self.feature_cols])
        self.current_regime = self.hmm.predict(train_regime[self.feature_cols].iloc[-1:])[0]
        
        X_3d = self._prepare_3d_features(X_train_scaled[self.feature_cols])
        y_3d = y_train.iloc[self.lookback-1:]
        
        if self.cnn is None:
            self.cnn = self._build_cnn(input_shape=(self.lookback, len(self.feature_cols)))
        
        self.cnn.fit(X_3d, y_3d + 1, epochs=10, batch_size=32)

    def _prepare_3d_features(self, data):
        sequences = []
        for i in range(len(data) - self.lookback + 1):
            seq = data.iloc[i:i+self.lookback]
            sequences.append(seq)
        return np.array(sequences)

    def predict(self, recent_data):
        current_regime = self.hmm.predict(recent_data[self.feature_cols].iloc[-self.lookback:])[-1]
        seq = self._prepare_3d_features(recent_data[self.feature_cols])[-1:]
        proba = self.cnn.predict(seq)[0]
        return np.argmax(proba) - 1  # Convert back to [-1,0,1]

# Train the model
model = SelfLearningTradingModel(n_regimes=3, cnn_lookback=32, feature_cols=X_train.columns.tolist())
model.train(X_train, y_train, train_regime, scaler)

# Test the model
X_test_3d = model._prepare_3d_features(X_test[model.feature_cols])
y_pred = np.argmax(model.cnn.predict(X_test_3d), axis=1) - 1
test_dates = X_test.index[model.lookback-1:]

# Evaluation of the model
from sklearn.metrics import classification_report
print(classification_report(y_test[model.lookback-1:], y_pred))
print("done")
