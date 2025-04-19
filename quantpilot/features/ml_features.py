from typing import List
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def plot_hmm_aic_bic(ori_df: pd.DataFrame, feature_cols: list, max_components: int = 10, n_iter=2000):
    df = ori_df.copy()
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    X = df[feature_cols].dropna().values
    
    aic = []
    bic = []
    lls = []
    # start from 2
    ns = range(2, max_components + 1)
    for n in ns:
        best_ll = None
        best_model = None
        for i in range(10):
            h = hmm.GaussianHMM(n, n_iter=n_iter, covariance_type='full', random_state=42)
            
            h.fit(X)
            score = h.score(X)
            if not best_ll or best_ll < score:
                best_ll = score
                best_model = h
        aic.append(best_model.aic(X))
        bic.append(best_model.bic(X))
        lls.append(best_model.score(X))
        
    fig, ax = plt.subplots()
    ln1 = ax.plot(ns, aic, label="AIC", color="blue", marker="o")
    ln2 = ax.plot(ns, bic, label="BIC", color="green", marker="o")
    ax2 = ax.twinx()
    ln3 = ax2.plot(ns, lls, label="LL", color="orange", marker="o")

    ax.legend(handles=ax.lines + ax2.lines)
    ax.set_title("Using AIC/BIC for Model Selection")
    ax.set_ylabel("Criterion Value (lower is better)")
    ax2.set_ylabel("LL (higher is better)")
    ax.set_xlabel("Number of HMM Components")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()

    plt.show()
        

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

def plot_kmeans_clusters_elbow(
    ori_df: pd.DataFrame, 
    feature_cols: List[str], 
    max_clusters: int = 10,
):
    """
    Plot the elbow method for KMeans clustering to determine the optimal number of clusters.

    :param df: Input DataFrame.
    :param feature_cols: Columns to use for clustering.
    :param max_clusters: Maximum number of clusters to test.
    """
    df = ori_df.copy()
    # Standardize the features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    X = df[feature_cols].dropna().values
    
    # Fit KMeans and calculate inertia for different cluster sizes
    sum_of_squared_distances = []
    K = range(1, max_clusters + 1)
    for num_clusters in K:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)
        sum_of_squared_distances.append(kmeans.inertia_)
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Sum of squared distances/Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    

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