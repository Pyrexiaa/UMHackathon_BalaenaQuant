import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# def get_feature_importance(X: pd.DataFrame, y: pd.Series, k=10):
#     selector = SelectKBest(score_func=f_classif, k=k)
#     selector.fit(X, y)
#     selected = X.columns[selector.get_support()].tolist()
#     scores = selector.scores_
#     importance = dict(zip(X.columns, scores))
#     return sorted(importance.items(), key=lambda x: x[1], reverse=True)[:k]

# Feature importance
def plot_feature_importance(df: pd.DataFrame, 
                             target_col: str,
                             feature_cols: List[str],
                             method: str = 'f_regression',
                             top_n: int = 10,
                             figsize: Tuple[int, int] = (12, 8),
                             save_img_path: str = None):
    """Plot and print feature importance based on various methods.

    Args:
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
        save_img_path : str
            File path to save the plot image (e.g., 'output/feature_importance.png')
            
    Returns:

    """
    # Clean data
    clean_df = df[feature_cols + [target_col]].dropna()
    # clean_df = df[feature_cols + [target_col]]
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
    top_features = importance_df.head(top_n)

    # Print top features
    print("\nTop Features:")
    print(top_features.to_string(index=False))

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(title)
    plt.tight_layout()

    # Save plot if path is provided
    if save_img_path:
        plt.savefig(save_img_path)
        print(f"\nPlot saved to: {save_img_path}")
        
    return top_features['feature'].tolist()
        
# Feature selection
def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    return df[features]
