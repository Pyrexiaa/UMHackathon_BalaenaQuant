import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from sklearn.feature_selection import f_classif, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


def plot_feature_importance(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    method: str = 'f_regression',
    top_n: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    save_img_path: str = None
) -> List[str]:
    """
    Plot and print feature importance based on the selected method.

    :param df: DataFrame containing the features and target variable.
    :param target_col: Column name of the target variable.
    :param feature_cols: List of feature columns to consider for importance scoring.
    :param method: Method used for computing importance. Options are:
                   'mutual_info', 'f_regression', 'random_forest', or 'correlation'.
    :param top_n: Number of top features to display in the plot.
    :param figsize: Size of the matplotlib figure.
    :param save_img_path: Optional path to save the plot image (e.g., 'output/importance.png').
    :return: List of the top N most important feature names.
    """
    clean_df = df[feature_cols + [target_col]].dropna()
    X = clean_df[feature_cols]
    y = clean_df[target_col]

    # Calculate feature importance
    if method == 'mutual_info':
        importances = mutual_info_regression(X, y)
        title = "Feature Importance (Mutual Information)"
    elif method == 'f_regression':
        importances, _ = f_classif(X, y)
        title = "Feature Importance (F-Statistics)"
    elif method == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        title = "Feature Importance (Random Forest)"
    elif method == 'correlation':
        importances = np.abs(X.corrwith(y)).values
        title = "Feature Importance (Correlation)"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot top features
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    top_features = importance_df.head(top_n)

    print("\nTop Features:")
    print(top_features.to_string(index=False))

    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(title)
    plt.tight_layout()

    if save_img_path:
        plt.savefig(save_img_path)
        print(f"\nPlot saved to: {save_img_path}")

    return top_features['feature'].tolist()


def select_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Select specific columns (features) from a DataFrame.

    :param df: Input DataFrame.
    :param features: List of feature column names to keep.
    :return: DataFrame containing only the selected features.
    """
    return df[features]