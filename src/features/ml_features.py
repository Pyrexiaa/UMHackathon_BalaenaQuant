from typing import Optional, List
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from .base_feature import BaseFeature

SENTIMENT_FILE_PATH = 'src/features/data/bitcoin_sentiments_21_24.csv'

class HMM(BaseFeature):
    def __init__(self, feature_cols: List[str] = ["close", "volume"], n_components: int = 3, n_iter: int = 1000):
        self.feature_cols = feature_cols
        self.n_components = n_components
        self.n_iter = n_iter

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=42
        )
        model.fit(df[self.feature_cols])
        df = df.copy()
        df["hmm_state"] = model.predict(df[self.feature_cols])
        return df
    

class RollingKMeans(BaseFeature):
    def __init__(self, feature_cols: List[str] = ["close", "volume"], n_clusters: int = 3, window_size: Optional[int] = None):
        self.feature_cols = feature_cols
        self.n_clusters = n_clusters
        self.window_size = window_size

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["kmeans_cluster"] = np.nan

        if self.window_size is None:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            df["kmeans_cluster"] = kmeans.fit_predict(df[self.feature_cols])
        else:
            for i in range(self.window_size, len(df)):
                window = df.iloc[i - self.window_size:i][self.feature_cols]
                if window.isnull().values.any():
                    continue
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                labels = kmeans.fit_predict(window.values)
                df.at[df.index[i - 1], "kmeans_cluster"] = labels[-1]
        return df


class NLPSentiment(BaseFeature):
    def __init__(self, sentiment_file_path: str = SENTIMENT_FILE_PATH, datetime_col: str = 'datetime'):
        self.sentiment_file_path = sentiment_file_path
        self.datetime_col = datetime_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        sentiment_df = pd.read_csv(self.sentiment_file_path)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        sentiment_df['Hour'] = sentiment_df['Date'].dt.floor('h')

        sentiment_dict = (
            sentiment_df.groupby('Hour')['Accurate Sentiments'].mean()
            .to_dict()
        )

        result_df = df.copy()

        if self.datetime_col is None or self.datetime_col == df.index.name:
            result_df['sentiment'] = result_df.index.map(lambda t: sentiment_dict.get(t.floor('H'), 0))
        else:
            if not pd.api.types.is_datetime64_any_dtype(result_df[self.datetime_col]):
                result_df[self.datetime_col] = pd.to_datetime(result_df[self.datetime_col])
            result_df['sentiment'] = result_df[self.datetime_col].dt.floor('h').map(sentiment_dict).fillna(0)

        return result_df