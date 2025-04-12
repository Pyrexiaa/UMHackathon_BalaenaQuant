# 🧠 Feature Engineering Module

The `feature_engineering` module provides a modular and extensible framework for extracting powerful features from financial time series data. It supports traditional technical indicators, machine learning-based feature generation, and a flexible base interface for building custom features.

---

## 📦 Module Structure

```
feature_engineering/
│
├── base_feature.py              # Abstract BaseFeature class
├── technical_indicators.py     # Technical indicator features (SMA, RSI, MACD, etc.)
└── ml_features.py              # ML-driven features (HMM states, KMeans clusters)
```

### BaseFeature Class `base_feature.py`

Key Methods:

-   `transform(df)` – Validates and applies the feature to a DataFrame.

-   `add_features(df)` – Abstract method to be implemented in subclasses.

-   `generate_feature_name()` – Auto-generates a default feature name based on class and window.

-   `clean_data()` – Ensures proper column input and datetime index for time-based features.

#### Usage Example

```py
class MovingAverageFeature(BaseFeature):
    def add_features(self, df):
        df[self.feature_name] = df[self.column].rolling(self.window).mean()
        return df
```

### Technical Indicators `(technical_indicators.py)`

Core Methods:

-   `add_sma(windows=[50, 200])` – Simple Moving Averages for crossover strategies.

-   `add_ema(windows=[5, 8, 13])` – EMAs with 5-8-13 crossover signals.

-   `add_rsi(windows=[14])` – Relative Strength Index with OBV confirmation signals.

-   `add_macd()` – MACD with signal crossovers and trade signals.

-   `add_price_change(windows=[1])` – Daily % change.

-   `add_volatility(windows=[24, 72, 168])` – Rolling volatility via log returns.

-   `add_bollinger_bands(windows=[20])` – Upper/lower band breakout signals.

-   `add_all_features()` – Apply all available indicators at once.

#### Usage Example

```py
fti = FeatureTechnicalIndicators(df, price_col="close")
fti.add_all_features()
```

### 🤖 Machine Learning Features `(ml_features.py)`

#### `add_hmm_features()`

Generates hidden state features using a Gaussian Hidden Markov Model (HMM) based on selected input features.

| Parameters            | Description                       |
| --------------------- | --------------------------------- |
| feature_cols          | Columns used to train HMM.        |
| n_components          | Number of hidden states.          |
| pretrained_model_path | Optional model to load from disk. |
| save_model_path       | Optional path to save the model.  |

Returns:
DataFrame with hmm_state column added.

#### `add_rolling_kmeans_cluster_feature()`

Applies KMeans clustering on rolling windows to generate regime/cluster labels.

| Parameters   | Description                                |
| ------------ | ------------------------------------------ |
| feature_cols | Columns to cluster on.                     |
| n_clusters   | Number of clusters.                        |
| window_size  | Size of the rolling window for clustering. |

Returns:
DataFrame with kmeans_cluster labels.

## 🧩 Feature Selection & Importance Module

This module provides tools for identifying and selecting important features from a dataset. It supports multiple statistical and model-based methods, as well as sentiment mapping for financial time series data.

---

### 📈 `plot_feature_importance()`

Visualizes the most important features using one of several techniques:

| Parameters      | Type              | Description                                                                              |
| --------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `df`            | `pd.DataFrame`    | Data containing both features and target                                                 |
| `target_col`    | `str`             | Name of the target variable                                                              |
| `feature_cols`  | `List[str]`       | List of feature column names                                                             |
| `method`        | `str`             | Importance method: `'mutual_info'`, `'f_regression'`, `'random_forest'`, `'correlation'` |
| `top_n`         | `int`             | Number of top features to show                                                           |
| `figsize`       | `Tuple[int, int]` | Size of the plot figure                                                                  |
| `save_img_path` | `str`             | Optional path to save the image output                                                   |

### Returns:

-   A list of top `n` most important feature names.

### Example:

```py
top_features = plot_feature_importance(
    df=data,
    target_col="future_return",
    feature_cols=feature_list,
    method="mutual_info",
    top_n=15,
    save_img_path="output/feature_importance.png"
)
```

### select_features()

Select a subset of features from a DataFrame.
|Parameters | Type| Description|
|-----------------|--------------------|-------------|
|df | pd.DataFrame | Input dataset|
|features |List[str] |List of feature names to retain|

### map_sentiments_to_hourly()

Maps external sentiment scores (e.g., from a CSV file) to the main dataset on an hourly basis. Useful for incorporating social or news-based signals.

| Argument            | Type         | Description                                                  |
| ------------------- | ------------ | ------------------------------------------------------------ |
| main_df             | pd.DataFrame | Time series DataFrame with datetime index or column          |
| sentiment_file_path | str          | Path to the sentiment CSV file                               |
| datetime_col        | str          | Name of the datetime column in main_df (default: "datetime") |

Returns: A copy of the main DataFrame with an added sentiment column.

Example:

```py
df_with_sentiment = map_sentiments_to_hourly(
    main_df=price_data,
    sentiment_file_path="data/sentiments.csv",
    datetime_col="datetime"
)
```

### Supported Methods for Feature Importance

| Method  | Description |
| ------------- | ---------------|
| mutual_info   | Uses mutual information regression to score features based on dependency with the target |
| f_regression  | Calculates F-statistics for each feature against the target                              |
| random_forest | Uses a RandomForestRegressor to derive feature importances                               |
| correlation   | Computes absolute Pearson correlation with the target                                    |
