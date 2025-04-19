# 🧠 Feature Selection Module

The `feature_selection` module is designed to extract the most relevant features from raw API-collected data to enhance the performance of a signals prediction model. It includes three key methods for statistical testing and rolling window optimization.

---

## 📦 Module Structure

```
feature_selection/
│
└── main.py                         # Feature Selection Class
```

### Feature Selection Class `main.py`

Core Methods:

-   `statistical_test(df, plot_dir)` – Performs feature-wise ANOVA tests to evaluate statistical significance across three target groups: sell, hold, and buy. It also calculates Cohen’s D and percentage difference. The method generates box plots for visual comparison and saves them to the specified directory.

-   `select_best_rolling_window_for_each_feature(df, plot_dir, rolling_windows)` – Evaluates multiple rolling window sizes for each feature. For every feature-window combination, it calculates the mean ANOVA p-value and ranks them accordingly to identify the most statistically significant transformations.

-   `select_best_rolling_window_for_all_features(df, feature_list)` – Determines the optimal rolling window size across all features by selecting the most frequently occurring window that yields the lowest ANOVA p-values.

### Usage Example

```py
class FeatureSelecion:
    def statistical_test(self, df, plot_dir="anova_plots"):
        ...
        return anova_results
    def select_best_rolling_window_for_each_feature(
        self, df, plot_dir="plots", rolling_windows=[24, 48, 72, 96, 120]
    ):
        ...
        return anova_df
    def select_best_rolling_window_for_all_features(self, df, feature_list):
        ...
        return most_common_window
```


#### 📈  `statistical_test()`

Performs feature-wise ANOVA tests to evaluate statistical significance across three target groups: sell, hold, and buy. It also calculates Cohen’s D and percentage difference. The method generates box plots for visual comparison and saves them to the specified directory.

| Parameters            | Description                       |
| --------------------- | --------------------------------- |
| df                    | Dataframe containing all features.|
| plot_dir              | The directory to save the plots.  |

Returns:
A list of ANOVA test results for each feature.

#### 📈  `select_best_rolling_window_for_each_feature()`

Evaluates multiple rolling window sizes for each feature. For every feature-window combination, it calculates the mean ANOVA p-value and ranks them accordingly to identify the most statistically significant transformations.

| Parameters            | Description                       |
| --------------------- | --------------------------------- |
| df                    | Dataframe containing all features.|
| plot_dir              | The directory to save the plots.  |
| rolling_windows       | Size of the rolling window.       |

Returns:
A DataFrame with p-values for each feature across all window sizes.

#### 📈 `select_best_rolling_window_for_all_features()`

Determines the optimal rolling window size across all features by selecting the most frequently occurring window that yields the lowest ANOVA p-values.

| Parameters      | Type              | Description                                                                              |
| --------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `df`            | `pd.DataFrame`    | Dataframe containing all features.                                                       |
| `feature_list`  | `list`            | A list of features to select from.                                                       |

Returns:
An integer representing the best global rolling window size.
