#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryptoAlphaLab - Feature Engineering Module

This module provides functions for generating advanced features for cryptocurrency trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import warnings
from hmmlearn import hmm

def add_technical_indicators(df: pd.DataFrame, 
                           timeframe: str = '1d',
                           custom_windows: Optional[Dict[str, List[int]]] = None,
                           add_sma: bool = True,
                           add_ema: bool = True,
                           add_rsi: bool = True,
                           add_macd: bool = True,
                           add_bbands: bool = True,
                           add_volatility: bool = True,
                           add_volume_indicators: bool = True,
                           add_price_channels: bool = True,
                           add_momentum: bool = True) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe based on specified timeframe.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing price data
    timeframe : str
        Time interval of data. Options: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
    custom_windows : Dict[str, List[int]], optional
        Custom window sizes for different indicators
        Example: {'sma': [5, 10, 20], 'rsi': [7, 14, 21]}
        If None, default windows will be used based on timeframe.
    add_sma : bool
        Whether to add Simple Moving Averages
    add_ema : bool
        Whether to add Exponential Moving Averages
    add_rsi : bool
        Whether to add Relative Strength Index
    add_macd : bool
        Whether to add Moving Average Convergence Divergence
    add_bbands : bool
        Whether to add Bollinger Bands
    add_volatility : bool
        Whether to add volatility indicators
    add_volume_indicators : bool
        Whether to add volume-based indicators
    add_price_channels : bool
        Whether to add price channel indicators
    add_momentum : bool
        Whether to add momentum indicators
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Map timeframe to appropriate window sizes
    timeframe_window_map = {
        '1m': {'short': [5, 10, 15], 'medium': [30, 60], 'long': [120, 240]}, 
        '5m': {'short': [3, 6, 12], 'medium': [24, 48], 'long': [96, 192]},
        '15m': {'short': [4, 8], 'medium': [16, 32], 'long': [64, 96]},
        '30m': {'short': [4, 8], 'medium': [16, 24], 'long': [48, 96]},
        '1h': {'short': [6, 12], 'medium': [24, 48], 'long': [72, 168]},
        '4h': {'short': [3, 6], 'medium': [12, 24], 'long': [36, 72]},
        '1d': {'short': [5, 10, 20], 'medium': [50, 100], 'long': [200]},
        '1w': {'short': [4, 8], 'medium': [26], 'long': [52]}
    }
    
    # Use custom windows if provided, otherwise use the mapped windows
    if not custom_windows:
        if timeframe not in timeframe_window_map:
            warnings.warn(f"Unknown timeframe: {timeframe}. Using default daily windows.")
            windows = timeframe_window_map['1d']
        else:
            windows = timeframe_window_map[timeframe]
    else:
        windows = custom_windows
    
    price_col = 'close'
    open_col = 'open'
    high_col = 'high'
    low_col = 'low'
    volume_col = 'volume'
    
    # 1. Price-based Indicators
    # ==============================
    
    # Simple Moving Averages (SMAs)
    if add_sma:
        sma_windows = windows.get('sma', windows['short'] + windows['medium'] + windows['long'])
        for window in sma_windows:
            result_df[f'sma_{window}'] = result_df[price_col].rolling(window=window).mean()
            # Calculate SMA crossovers and divergence
            if window in windows['short'] and windows['medium']:
                med_window = windows['medium'][0]
                result_df[f'sma_cross_{window}_{med_window}'] = np.where(
                    result_df[f'sma_{window}'] > result_df[f'sma_{med_window}'], 1, -1)
    
    # Exponential Moving Averages (EMAs)
    if add_ema:
        ema_windows = windows.get('ema', windows['short'] + windows['medium'])
        for window in ema_windows:
            result_df[f'ema_{window}'] = result_df[price_col].ewm(span=window, adjust=False).mean()
        
        # Calculate EMA crossovers
        if len(ema_windows) >= 2:
            short_ema = ema_windows[0]
            long_ema = ema_windows[-1]
            result_df[f'ema_cross_{short_ema}_{long_ema}'] = np.where(
                result_df[f'ema_{short_ema}'] > result_df[f'ema_{long_ema}'], 1, -1)
    
    # Relative Strength Index (RSI)
    if add_rsi:
        rsi_windows = windows.get('rsi', [14])
        for window in rsi_windows:
            delta = result_df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # Calculate RSI
            rs = gain / loss
            result_df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # Add RSI crossover signals
            result_df[f'rsi_{window}_overbought'] = np.where(result_df[f'rsi_{window}'] > 70, 1, 0)
            result_df[f'rsi_{window}_oversold'] = np.where(result_df[f'rsi_{window}'] < 30, 1, 0)
    
    # MACD (Moving Average Convergence Divergence)
    if add_macd:
        macd_params = windows.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
        fast = macd_params['fast']
        slow = macd_params['slow']
        signal_period = macd_params['signal']
        
        # Calculate MACD components
        ema_fast = result_df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = result_df[price_col].ewm(span=slow, adjust=False).mean()
        result_df['macd_line'] = ema_fast - ema_slow
        result_df['macd_signal'] = result_df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        result_df['macd_histogram'] = result_df['macd_line'] - result_df['macd_signal']
        
        # MACD crossover signal
        result_df['macd_crossover'] = np.where(
            result_df['macd_line'] > result_df['macd_signal'], 1, -1)
    
    # Bollinger Bands
    if add_bbands:
        bb_windows = windows.get('bbands', [20])
        for window in bb_windows:
            # Calculate Bollinger Bands
            result_df[f'bb_middle_{window}'] = result_df[price_col].rolling(window=window).mean()
            result_df[f'bb_std_{window}'] = result_df[price_col].rolling(window=window).std()
            result_df[f'bb_upper_{window}'] = result_df[f'bb_middle_{window}'] + 2 * result_df[f'bb_std_{window}']
            result_df[f'bb_lower_{window}'] = result_df[f'bb_middle_{window}'] - 2 * result_df[f'bb_std_{window}']
            
            # Add BB width and %B indicator
            result_df[f'bb_width_{window}'] = (result_df[f'bb_upper_{window}'] - result_df[f'bb_lower_{window}']) / result_df[f'bb_middle_{window}']
            result_df[f'bb_pct_b_{window}'] = (result_df[price_col] - result_df[f'bb_lower_{window}']) / (result_df[f'bb_upper_{window}'] - result_df[f'bb_lower_{window}'])
            
            # BB signals
            result_df[f'bb_upper_cross_{window}'] = np.where(result_df[price_col] > result_df[f'bb_upper_{window}'], 1, 0)
            result_df[f'bb_lower_cross_{window}'] = np.where(result_df[price_col] < result_df[f'bb_lower_{window}'], 1, 0)
    
    # 2. Volatility Indicators
    # ==============================
    if add_volatility and high_col in df.columns and low_col in df.columns:
        # ATR (Average True Range)
        atr_windows = windows.get('volatility', [14]) 
        
        for window in atr_windows:
            # True Range
            high_low = result_df[high_col] - result_df[low_col]
            high_close = np.abs(result_df[high_col] - result_df[price_col].shift())
            low_close = np.abs(result_df[low_col] - result_df[price_col].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            # Average True Range (ATR)
            result_df[f'atr_{window}'] = true_range.rolling(window).mean()
            
            # ATR percent
            result_df[f'atr_pct_{window}'] = result_df[f'atr_{window}'] / result_df[price_col] * 100
            
            # Historical Volatility (close-to-close)
            ln_returns = np.log(result_df[price_col] / result_df[price_col].shift(1))
            result_df[f'volatility_{window}'] = ln_returns.rolling(window).std() * np.sqrt(252)  # Annualized
    
    # Price change
    result_df['price_change_1'] = result_df[price_col].pct_change()
    result_df['price_change_5'] = result_df[price_col].pct_change(periods=5)
    result_df['price_change_10'] = result_df[price_col].pct_change(periods=10)
    
    # 3. Volume Indicators
    # ==============================
    if add_volume_indicators and volume_col and volume_col in df.columns:
        vol_windows = windows.get('volume', [5, 20])
        
        # Volume SMAs
        for window in vol_windows:
            result_df[f'volume_sma_{window}'] = result_df[volume_col].rolling(window=window).mean()
        
        # Volume Oscillator
        if len(vol_windows) >= 2:
            short_vol = vol_windows[0]
            long_vol = vol_windows[-1]
            result_df['volume_oscillator'] = (
                (result_df[f'volume_sma_{short_vol}'] - result_df[f'volume_sma_{long_vol}']) / 
                result_df[f'volume_sma_{long_vol}'] * 100
            )
        
        # Money Flow Volume (MFV)
        typical_price = (result_df[high_col] + result_df[low_col] + result_df[price_col]) / 3
        raw_money_flow = typical_price * result_df[volume_col]
        
        # Money Flow Direction
        money_flow_direction = np.where(typical_price > typical_price.shift(1), 1, -1)
        money_flow = raw_money_flow * money_flow_direction
        
        # Money Flow Index
        mfi_period = windows.get('mfi', [14])[0]
        positive_flow = money_flow.where(money_flow > 0, 0).rolling(window=mfi_period).sum()
        negative_flow = abs(money_flow.where(money_flow < 0, 0)).rolling(window=mfi_period).sum()
        
        money_ratio = positive_flow / negative_flow
        result_df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # On-Balance Volume (OBV)
        obv = (np.sign(result_df[price_col].diff()) * result_df[volume_col]).fillna(0)
        result_df['obv'] = obv.cumsum()
        
        # Chaikin Money Flow
        cmf_period = windows.get('cmf', [20])[0]
        money_flow_volume = ((result_df[price_col] - result_df[low_col]) - (result_df[high_col] - result_df[price_col])) / (result_df[high_col] - result_df[low_col]) * result_df[volume_col]
        result_df['cmf'] = money_flow_volume.rolling(cmf_period).sum() / result_df[volume_col].rolling(cmf_period).sum()
    
    # 4. Price Channels
    # ==============================
    if add_price_channels:
        pc_windows = windows.get('price_channels', [20])
        
        for window in pc_windows:
            # Donchian Channels
            result_df[f'upper_channel_{window}'] = result_df[high_col].rolling(window=window).max()
            result_df[f'lower_channel_{window}'] = result_df[low_col].rolling(window=window).min()
            result_df[f'middle_channel_{window}'] = (result_df[f'upper_channel_{window}'] + result_df[f'lower_channel_{window}']) / 2
            
            # Breakout signals
            result_df[f'upper_breakout_{window}'] = np.where(result_df[price_col] > result_df[f'upper_channel_{window}'].shift(1), 1, 0)
            result_df[f'lower_breakout_{window}'] = np.where(result_df[price_col] < result_df[f'lower_channel_{window}'].shift(1), 1, 0)
    
    # 5. Momentum Indicators
    # ==============================
    if add_momentum:
        # Rate of Change
        roc_periods = windows.get('roc', [5, 10, 20])
        for period in roc_periods:
            result_df[f'roc_{period}'] = (
                (result_df[price_col] - result_df[price_col].shift(period)) / 
                result_df[price_col].shift(period) * 100
            )
        
        # Stochastic Oscillator
        stoch_period = windows.get('stochastic', [14])[0]
        result_df['stoch_k'] = 100 * ((result_df[price_col] - result_df[low_col].rolling(window=stoch_period).min()) / 
                                     (result_df[high_col].rolling(window=stoch_period).max() - 
                                      result_df[low_col].rolling(window=stoch_period).min()))
        result_df['stoch_d'] = result_df['stoch_k'].rolling(window=3).mean()
        
        # Commodity Channel Index (CCI)
        cci_period = windows.get('cci', [20])[0]
        typical_price = (result_df[high_col] + result_df[low_col] + result_df[price_col]) / 3
        mean_dev = np.abs(typical_price - typical_price.rolling(window=cci_period).mean()).rolling(window=cci_period).mean()
        result_df['cci'] = (typical_price - typical_price.rolling(window=cci_period).mean()) / (0.015 * mean_dev)
    
    return result_df


def add_hmm_features(df: pd.DataFrame, 
                   price_col: str = 'close_price',
                   returns_col: Optional[str] = None,
                   n_states: int = 3,
                   feature_cols: Optional[List[str]] = None,
                   lookback_periods: List[int] = [5, 10, 20],
                   normalize: bool = True) -> pd.DataFrame:
    """
    Add Hidden Markov Model derived features to detect market regimes.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing price data
    price_col : str
        Column name for close price
    returns_col : str, optional
        Column name for returns, if None, returns will be calculated from price_col
    n_states : int
        Number of regimes (states) to detect
    feature_cols : List[str], optional
        Columns to use as features for HMM (if None, returns will be used)
    lookback_periods : List[int]
        Periods for probability smoothing
    normalize : bool
        Whether to normalize input features
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with added HMM features:
        - Regime states
        - Regime probabilities
        - Smoothed probabilities
    """
    result_df = df.copy()
    
    # Calculate returns if not provided
    if returns_col is None:
        result_df['returns'] = result_df[price_col].pct_change()
        returns_col = 'returns'
    
    # Prepare features for HMM
    if feature_cols is None:
        # Use returns as the only feature by default
        X = result_df[returns_col].dropna().values.reshape(-1, 1)
    else:
        # Use multiple features if specified
        X = result_df[feature_cols].dropna().values
    
    # Skip if not enough data
    if len(X) < 50:
        warnings.warn("Insufficient data for HMM modeling. At least 50 rows required.")
        return result_df
    
    # Normalize features if requested
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Fit HMM model
    model = hmm.GaussianHMM(
        n_components=n_states, 
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    
    try:
        model.fit(X)
        
        # Create mapping for all rows in the original dataframe
        regime = np.full(len(result_df), np.nan)
        regime_probs = np.zeros((len(result_df), n_states))
        
        # Find indices where we have valid features
        valid_indices = result_df.index[~result_df[returns_col if feature_cols is None else feature_cols[0]].isna()]
        
        if len(valid_indices) > 0:
            # Predict regimes and probabilities
            valid_states = model.predict(X)
            valid_probs = model.predict_proba(X)
            
            # Fill the arrays at valid indices
            regime[valid_indices] = valid_states
            regime_probs[valid_indices] = valid_probs
            
            # Add regime state to the dataframe
            result_df['hmm_regime'] = regime
            
            # Add regime probabilities
            for i in range(n_states):
                result_df[f'hmm_prob_{i}'] = regime_probs[:, i]
                
                # Add smoothed probabilities
                for period in lookback_periods:
                    result_df[f'hmm_prob_{i}_smooth_{period}'] = result_df[f'hmm_prob_{i}'].rolling(window=period).mean()
            
            # Add regime change signals
            result_df['hmm_regime_change'] = result_df['hmm_regime'].diff().abs() > 0
            
            # Add regime duration
            result_df['hmm_regime_duration'] = 0
            for regime_id in range(n_states):
                mask = result_df['hmm_regime'] == regime_id
                # Count consecutive occurrences of this regime
                if mask.any():
                    result_df.loc[mask, 'hmm_regime_duration'] = mask.astype(int).groupby((~mask).cumsum()).cumsum()
            
            # Analyze regimes and add labels
            regime_stats = {}
            for i in range(n_states):
                regime_returns = result_df.loc[result_df['hmm_regime'] == i, returns_col].dropna()
                if len(regime_returns) > 0:
                    regime_stats[i] = {
                        'mean': regime_returns.mean(),
                        'std': regime_returns.std(),
                        'count': len(regime_returns)
                    }
            
            # Label regimes by average return (bearish, neutral, bullish)
            if len(regime_stats) == n_states:
                regime_by_return = sorted(regime_stats.keys(), key=lambda k: regime_stats[k]['mean'])
                
                regime_labels = {}
                if n_states == 2:
                    regime_labels = {
                        regime_by_return[0]: 'bearish',
                        regime_by_return[1]: 'bullish'
                    }
                elif n_states == 3:
                    regime_labels = {
                        regime_by_return[0]: 'bearish',
                        regime_by_return[1]: 'neutral',
                        regime_by_return[2]: 'bullish'
                    }
                else:
                    # More than 3 states
                    for i, r in enumerate(regime_by_return):
                        if i == 0:
                            regime_labels[r] = 'strongly_bearish'
                        elif i == n_states - 1:
                            regime_labels[r] = 'strongly_bullish'
                        elif i < n_states // 2:
                            regime_labels[r] = f'bearish_{i}'
                        elif i > n_states // 2:
                            regime_labels[r] = f'bullish_{i-n_states//2}'
                        else:
                            regime_labels[r] = 'neutral'
                
                # Add regime label column
                result_df['hmm_regime_label'] = result_df['hmm_regime'].map(regime_labels)
                
                # Add a numeric score for the regime (-1 to 1)
                min_regime = min(regime_by_return)
                max_regime = max(regime_by_return)
                if min_regime != max_regime:
                    result_df['hmm_regime_score'] = result_df['hmm_regime'].apply(
                        lambda x: 2 * (x - min_regime) / (max_regime - min_regime) - 1 if not pd.isna(x) else np.nan
                    )
        
    except Exception as e:
        warnings.warn(f"Error in HMM modeling: {str(e)}")
    
    return result_df


def add_cnn_features(df: pd.DataFrame,
                   price_col: str = 'close_price',
                   window_size: int = 30,
                   n_filters: List[int] = [8, 16, 16, 8],
                   filter_sizes: List[int] = [3, 5, 5, 3],
                   n_features: int = 8,
                   lookback_periods: List[int] = [1, 5, 10],
                   use_pretrained: bool = False,
                   model_path: Optional[str] = None) -> pd.DataFrame:
    """
    Add Convolutional Neural Network derived features for pattern recognition.
    
    This function uses a CNN autoencoder to extract patterns from price movements
    and creates features that capture these patterns. The CNN learns to recognize
    complex patterns that may not be visible through standard technical indicators.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing price data
    price_col : str
        Column name for close price
    window_size : int
        Size of the sliding window for pattern recognition
    n_filters : List[int]
        Number of filters in each convolutional layer
    filter_sizes : List[int]
        Size of filters in each convolutional layer
    n_features : int
        Number of CNN-derived features to generate
    lookback_periods : List[int]
        Periods for calculating changes in CNN features
    use_pretrained : bool
        Whether to use a pre-trained model
    model_path : str, optional
        Path to the pre-trained model file
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with added CNN-derived features
    """
    result_df = df.copy()
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape, Lambda
    except ImportError:
        warnings.warn("TensorFlow not installed. Cannot create CNN features.")
        return result_df
    
    # Skip if not enough data
    if len(df) < window_size * 2:
        warnings.warn(f"Insufficient data for CNN features. Need at least {window_size * 2} rows.")
        return result_df
    
    # Normalize price data for better learning
    price_series = result_df[price_col].values
    price_scaler = StandardScaler()
    price_scaled = price_scaler.fit_transform(price_series.reshape(-1, 1)).flatten()
    
    # Create overlapping windows for training
    windows = []
    for i in range(len(price_scaled) - window_size + 1):
        window = price_scaled[i:i+window_size]
        windows.append(window)
    
    if not windows:
        warnings.warn("No valid windows created.")
        return result_df
    
    windows = np.array(windows).reshape(-1, window_size, 1)
    
    if use_pretrained and model_path:
        # Load pre-trained model
        try:
            autoencoder = load_model(model_path)
            encoder = Model(inputs=autoencoder.input, 
                           outputs=autoencoder.get_layer(f'encoded').output)
        except Exception as e:
            warnings.warn(f"Could not load pre-trained model: {str(e)}")
            use_pretrained = False
    
    if not use_pretrained:
        # Build CNN autoencoder model
        input_window = Input(shape=(window_size, 1))
        
        # Encoder
        encoded = input_window
        for i, (n_filter, filter_size) in enumerate(zip(n_filters, filter_sizes)):
            encoded = Conv1D(n_filter, filter_size, activation='relu', padding='same')(encoded)
            if i < len(n_filters) - 1:  # No pooling on the last layer
                encoded = MaxPooling1D(2, padding='same')(encoded)
        
        # Bottleneck - extract encoded features
        encoded = Conv1D(n_features, 1, activation='relu', name='encoded')(encoded)
        
        # Create and train the model
        autoencoder = Model(input_window, encoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train with early stopping
        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='loss', patience=3)
        
        autoencoder.fit(
            windows, windows,
            epochs=50,
            batch_size=32,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Extract encoder part
        encoder = Model(inputs=autoencoder.input, 
                       outputs=autoencoder.get_layer('encoded').output)
    
    # Generate CNN features for all windows
    cnn_features = encoder.predict(windows)
    
    # Average features across the time dimension to get a single value per window
    cnn_features_avg = np.mean(cnn_features, axis=1)
    
    # Create dataframe with CNN features
    cnn_df = pd.DataFrame(
        cnn_features_avg, 
        columns=[f'cnn_feature_{i}' for i in range(cnn_features_avg.shape[1])]
    )
    
    # Add timestep index
    cnn_df.index = df.index[window_size-1:]
    
    # Merge CNN features with original dataframe
    result_df = pd.merge(
        result_df, cnn_df, 
        left_index=True, right_index=True, 
        how='left'
    )
    
    # Add feature changes over different periods
    for feature in cnn_df.columns:
        for period in lookback_periods:
            result_df[f'{feature}_change_{period}'] = result_df[feature].diff(period)
    
    # Add CNN pattern clusters
    # Use K-means to cluster patterns and add as features
    try:
        from sklearn.cluster import KMeans
        
        n_clusters = min(5, len(cnn_features_avg))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(cnn_features_avg)
        
        # Add cluster label to dataframe
        cluster_series = pd.Series(clusters, index=result_df.index[window_size-1:])
        result_df['cnn_pattern_cluster'] = cluster_series
        
        # Add one-hot encoded clusters
        for i in range(n_clusters):
            result_df[f'cnn_cluster_{i}'] = (result_df['cnn_pattern_cluster'] == i).astype(float)
            
        # Add cluster transitions (change in cluster)
        result_df['cnn_cluster_change'] = result_df['cnn_pattern_cluster'].diff().abs() > 0
        
    except Exception as e:
        warnings.warn(f"Could not create CNN pattern clusters: {str(e)}")
    
    return result_df

def select_features(df: pd.DataFrame, 
                   target_col: str,
                   feature_cols: Optional[List[str]] = None,
                   method: str = 'mutual_info',
                   n_features: int = 20,
                   corr_threshold: float = 0.85) -> List[str]:
    """
    Select the most important features based on their relationship with the target variable.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing features and target
    target_col : str
        Column name of the target variable
    feature_cols : List[str], optional
        List of potential feature columns to consider (if None, all numeric columns except target)
    method : str
        Feature selection method
        Options: 'mutual_info', 'f_regression', 'random_forest', 'correlation'
    n_features : int
        Number of features to select
    corr_threshold : float
        Correlation threshold for removing highly correlated features
        
    Returns:
    -------
    List[str]
        List of selected feature column names
    """
    # Make sure the target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Identify feature columns if not provided
    if feature_cols is None:
        # Use all numeric columns except the target
        feature_cols = [col for col in df.select_dtypes(include=np.number).columns 
                        if col != target_col]
    
    # Clean data for feature selection
    clean_df = df[feature_cols + [target_col]].dropna()
    
    if len(clean_df) < 10:
        warnings.warn("Not enough clean data for feature selection")
        return feature_cols[:min(n_features, len(feature_cols))]
    
    X = clean_df[feature_cols]
    y = clean_df[target_col]
    
    # Calculate feature importance
    if method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        importances = mutual_info_regression(X, y)
    elif method == 'f_regression':
        from sklearn.feature_selection import f_regression
        importances, _ = f_regression(X, y)
    elif method == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
    elif method == 'correlation':
        # Use absolute correlation with target
        importances = np.abs(X.corrwith(y)).values
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top features by importance
    top_features = importance_df['feature'].tolist()[:n_features * 2]  # Select more for correlation filtering
    
    # Filter out highly correlated features
    selected_features = []
    
    if corr_threshold < 1.0:
        # Calculate correlation matrix for top features
        corr_matrix = X[top_features].corr().abs()
        
        # Add features one by one, avoiding high correlation with already selected
        for feature in top_features:
            if not selected_features:
                selected_features.append(feature)
            else:
                # Check correlation with already selected features
                corrs = corr_matrix.loc[feature, selected_features]
                if corrs.max() < corr_threshold:
                    selected_features.append(feature)
            
            # Stop when we have enough features
            if len(selected_features) >= n_features:
                break
    else:
        # No correlation filtering, just take top n_features
        selected_features = top_features[:n_features]
    
    return selected_features


# Optional: Feature importance visualization function
def plot_feature_importance(df: pd.DataFrame, 
                          target_col: str,
                          feature_cols: List[str],
                          method: str = 'random_forest',
                          top_n: int = 20,
                          figsize: Tuple[int, int] = (12, 8)):
    """
    Plot feature importance based on various methods.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing features and target
    target_col : str
        Column name of the target variable
    feature_cols : List[str]
        List of feature columns to consider
    method : str
        Feature importance method
        Options: 'mutual_info', 'f_regression', 'random_forest', 'correlation'
    top_n : int
        Number of top features to show
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    -------
    matplotlib.figure.Figure
        The feature importance plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Clean data
        clean_df = df[feature_cols + [target_col]].dropna()
        X = clean_df[feature_cols]
        y = clean_df[target_col]
        
        # Calculate feature importance
        if method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            importances = mutual_info_regression(X, y)
            title = "Feature Importance (Mutual Information)"
        elif method == 'f_regression':
            from sklearn.feature_selection import f_regression
            importances, _ = f_regression(X, y)
            title = "Feature Importance (F-Statistics)"
        elif method == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            title = "Feature Importance (Random Forest)"
        elif method == 'correlation':
            importances = np.abs(X.corrwith(y)).values
            title = "Feature Importance (Correlation)"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create dataframe for plotting
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Limit to top_n features
        if len(importance_df) > top_n:
            importance_df = importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(title)
        plt.tight_layout()
        
        return plt.gcf()
    
    except Exception as e:
        warnings.warn(f"Could not create feature importance plot: {str(e)}")
        return None