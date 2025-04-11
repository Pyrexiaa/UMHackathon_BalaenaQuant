import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def get_feature_importance(X: pd.DataFrame, y: pd.Series, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected = X.columns[selector.get_support()].tolist()
    scores = selector.scores_
    importance = dict(zip(X.columns, scores))
    return sorted(importance.items(), key=lambda x: x[1], reverse=True)[:k]

def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    return df[features]
