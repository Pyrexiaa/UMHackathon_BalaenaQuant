import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Add the path to your 'src' folder dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.feature_engineering import FeatureTechnicalIndicators, plot_feature_importance, select_features

# Load sample data
df = pd.read_csv('merged_features.csv', parse_dates=['datetime'], index_col='datetime')

# Display data info
print("\nData preview:")
print(df.head())

# 2. Basic technical indicators
print("\n2. Adding basic technical indicators...")
df_tech = df.copy()
feature_technical = FeatureTechnicalIndicators(df_tech, price_col='close')

# Add technical indicators
feature_technical.add_ema([5, 8, 13])
feature_technical.add_sma([5, 20])
feature_technical.add_rsi([14])
feature_technical.add_macd(fast_window=12, slow_window=26, signal_window=9)
feature_technical.add_price_change([1])

# Show added features by comparing with original DataFrame
added_features = list(set(df_tech.columns) - set(df.columns))
print(f"Added {len(added_features)} features: {', '.join(added_features)}")
print(df_tech.head())
print(df_tech[['close'] + added_features].head())


# 3. Feature selection
### NOTE: the target column should be the one you want to predict, i.e., 'buy' or 'sell' signal
### Here we just use 'close' as the target column for demonstration purposes
print("\n3. Feature selection...") 
df_tech = df_tech.dropna()
top10_tech = plot_feature_importance(df_tech, target_col='close', feature_cols=added_features, top_n=10, save_img_path='feature_importance.png', method='correlation')
print(f"Top 10 technical features: {top10_tech}")

X = df_tech.drop(columns=['close'])
top10_all = plot_feature_importance(df_tech, target_col='close', feature_cols=list(X.columns), top_n=10, save_img_path='feature_importance_all.png', method='f_regression')
print(f"Top 10 features (all): {top10_all}")

# Select features based on correlation
selected_features = select_features(df_tech, features=top10_tech + ['close'])
print(f"Selected features: {selected_features}")