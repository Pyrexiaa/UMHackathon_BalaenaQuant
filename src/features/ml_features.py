from typing import Optional, Union
from .base_feature import BaseFeature
import pandas as pd
import numpy as np
from hmmlearn import hmm
from typing import List
import pickle

# class HMMFeature(BaseFeature):
#     def __init__(self, model, column: str = "close", window: Optional[Union[int, str]] = None):
#         """
#         :param model: Pretrained HMM model that supports a `predict` method.
#         :param column: Column to use for feature extraction, default is 'close'.
#         :param window: Lookback window (int or str, e.g., '14d', '3h')
#         """
#         super().__init__(column, window)
#         self.model = model  # Assume model has a `predict` method

#     def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Add HMM feature to the DataFrame by predicting the hidden states
#         using the provided HMM model.
#         """
#         # Ensure the 'close' column exists
#         if self.column not in df.columns:
#             raise ValueError(f"Column '{self.column}' not found in DataFrame.")
        
#         # Model prediction (Assume model predicts hidden states based on `df[self.column]`)
#         hidden_states = self.model.predict(df[[self.column]])
        
#         # Add hidden states as a new feature (column)
#         df[self.feature_name] = hidden_states
#         return df

def add_hmm_features(ori_df: pd.DataFrame, 
                     feature_cols: List[str], 
                     n_components: int = 3,
                     n_iter: int = 1000,
                     save_model_path: str = None,
                     pretrained_model_path: Optional[str] = None) -> pd.DataFrame:
    """
    Add HMM-based hidden state features to a DataFrame.

    Args:
        ori_df : pd.DataFrame
            DataFrame containing the feature columns.
        feature_cols : List[str]
            List of columns to use as input for HMM (e.g., ['close', 'volume']).
        n_components : int
            Number of hidden states for the HMM.
        n_iter : int
            Number of iterations for HMM fitting.
        save_model_path : str
            Path to save the trained HMM model.
        pretrained_model_path : Optional[str]
            Path to a pretrained HMM model. If provided, the model will be loaded instead of trained.            
    Returns:
        df : pd.DataFrame
            DataFrame with an added HMM state column (or columns).
    """
    df = ori_df.copy()

    if pretrained_model_path:
        # Load the pretrained model
        with open(pretrained_model_path, 'rb') as f:
            hmm_model = pickle.load(f)

    else:  
        hmm_model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", 
                                n_iter=n_iter, random_state=42)  
        hmm_model.fit(df[feature_cols].values)
    
        # Save the model if a path is provided
        if save_model_path:
            with open(save_model_path, 'wb') as f:
                pickle.dump(hmm_model, f)
                
    hidden_states = hmm_model.predict(df[feature_cols].values)
    df['hmm_state'] = hidden_states
    
    return df
