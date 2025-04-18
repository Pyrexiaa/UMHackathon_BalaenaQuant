import pickle
from typing import Optional, Union, List

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans


def add_hmm_features(
    ori_df: pd.DataFrame, 
    feature_cols: List[str] = ["close", "volume"], 
    n_components: int = 3,
    n_iter: int = 1000,
    save_model_path: Optional[str] = None,
    pretrained_model_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Add HMM-based hidden state features to a DataFrame.

    :param ori_df: Input DataFrame.
    :param feature_cols: Columns to use as input for HMM (e.g., ['close', 'volume']).
    :param n_components: Number of hidden states for the HMM.
    :param n_iter: Number of iterations for HMM fitting.
    :param save_model_path: Optional path to save the trained HMM model.
    :param pretrained_model_path: Optional path to load a pretrained HMM model.
    :return: DataFrame with an added 'hmm_state' column.
    """
    df = ori_df.copy()

    if pretrained_model_path:
        with open(pretrained_model_path, 'rb') as f:
            hmm_model = pickle.load(f)
    else:  
        hmm_model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag", 
            n_iter=n_iter,
            random_state=42
        )
        hmm_model.fit(df[feature_cols].values)

        if save_model_path:
            with open(save_model_path, 'wb') as f:
                pickle.dump(hmm_model, f)

    hidden_states = hmm_model.predict(df[feature_cols].values)
    df['hmm_state'] = hidden_states

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

    return result_df