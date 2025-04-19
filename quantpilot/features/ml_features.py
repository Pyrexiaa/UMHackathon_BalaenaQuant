import pickle
from typing import Optional, Union, List

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def add_hmm_features(
    ori_df: pd.DataFrame, 
    feature_cols: List[str] = [
        "addresses_count_active",          
        "addresses_count_receiver",        
        "addresses_count_sender",          
        "fees_transaction_mean_usd",        
        "miner_supply_ratio",              
        "exchange_supply_ratio",            
        "transactions_count_inflow",       
        "transactions_count_outflow",      
        "tokens_transferred_total",        
        "exchange_whale_ratio",             
        "coinbase_premium_index_usdt_adjusted", 
        "coinbase_premium_gap_usdt_adjusted",  
        "long_liquidations_usd",               
        "short_liquidations_usd"               
    ], 
    n_components: int = 3,
    n_iter: int = 2000,
) -> pd.DataFrame:
    """
    Add HMM-based hidden state features to a DataFrame.

    :param ori_df: Input DataFrame.
    :param feature_cols: Columns to use as input for HMM (e.g., ['close', 'volume']).
    :param n_components: Number of hidden states for the HMM.
    :param n_iter: Number of iterations for HMM fitting.
    :return: DataFrame with an added 'hmm_state' column.
    """
    df = ori_df.copy()
    hmm_model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full", 
        n_iter=n_iter,
        random_state=42
    )
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    # Fit the HMM model
    hmm_model.fit(df[feature_cols].values)

    hidden_states = hmm_model.predict(df[feature_cols].values)
    df['hmm_state'] = hidden_states

    return df

def add_rolling_kmeans_cluster_feature(df: pd.DataFrame, 
                                       feature_cols: List[str], 
                                       cluster_col_name: str = 'kmeans_cluster', 
                                       n_clusters: int = 3,
                                       window_size: int = None) -> pd.DataFrame:
    """
    Add rolling KMeans cluster label based on selected feature columns.

    Parameters:
        df : pd.DataFrame
            Input DataFrame.
        feature_cols : List[str]
            Columns used for clustering.
        cluster_col_name : str
            Name of the new column to store the cluster label.
        n_clusters : int
            Number of clusters for KMeans.
        window_size : int
            Rolling window size for fitting KMeans.

    Returns:
        pd.DataFrame with a new column of rolling cluster labels.
    """
    df = df.copy()
    df[cluster_col_name] = np.nan  # Initialize output column
    # Standardize the features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    if window_size is None:
        # If no window size is provided, fit KMeans on the entire DataFrame
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(df[feature_cols].values)
        df[cluster_col_name] = labels
    else:
        for i in range(window_size, len(df)):
            window_data = df.iloc[i - window_size:i][feature_cols]
            if window_data.isnull().values.any():
                continue  # skip if NaN exists in window

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(window_data.values)

            # Assign last point's label to the current row
            df.at[df.index[i - 1], cluster_col_name] = labels[-1]

    return df

def add_nlp_sentiment_score(
    main_df: pd.DataFrame,
    sentiment_file_path: str = 'quantpilot/features/data/bitcoin_sentiments_21_24.csv',
    datetime_col: str = 'datetime'
) -> pd.DataFrame:
    """
    Maps hourly sentiment scores to a main DataFrame.

    :param main_df: Main DataFrame with hourly data, containing a datetime column or index.
    :param sentiment_file_path: Path to the sentiment CSV file with 'Date' and 'Accurate Sentiments' columns.
    :param datetime_col: Name of the datetime column in `main_df`. If None or matches index name, uses the index.
    :return: DataFrame with a new 'sentiment' column mapped by the hour.
    """
    sentiment_df = pd.read_csv(sentiment_file_path)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sentiment_df['Hour'] = sentiment_df['Date'].dt.floor('h')

    # Get the average sentiment for each hour
    hourly_sentiments = sentiment_df.groupby('Hour')['Accurate Sentiments'].mean().reset_index()
    sentiment_dict = dict(zip(hourly_sentiments['Hour'], hourly_sentiments['Accurate Sentiments']))

    result_df = main_df.copy()

    if datetime_col is None or datetime_col == main_df.index.name:
        result_df['sentiment'] = pd.Series(
            index=result_df.index,
            data=[sentiment_dict.get(idx, 0) for idx in result_df.index]
        )
    else:
        if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
            result_df[datetime_col] = pd.to_datetime(result_df[datetime_col])
        result_df['sentiment'] = result_df[datetime_col].map(sentiment_dict).fillna(0)

    # positive: sentiment score > 0
    # neutral: sentiment score = 0
    # negative: sentiment score < 0
    result_df['sentiment'] = result_df['sentiment'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    return result_df