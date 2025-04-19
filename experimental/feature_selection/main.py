import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from experimental.modeling.utils import load_csv
from scipy.stats import f_oneway
from itertools import combinations
from ..modeling.utils import interpret_cohens_d

from experimental.modeling.constants import ALL_ON_CHAIN_FEATURES, ASSUMPTION_10


class FeatureSelecion:
    def __init__(self):
        pass

    def statistical_test(self, df, plot_dir="anova_plots"):
        anova_results = []
        features = [f for f in ALL_ON_CHAIN_FEATURES if f in df.columns]
        for feature in features:
            if feature not in df.columns:
                continue

            target_values = sorted(df["target"].dropna().unique())
            groups = [df[df["target"] == tv][feature].dropna() for tv in target_values]

            if len(groups) > 1:
                f_val, p_val = f_oneway(*groups)
                anova_results.append(
                    {"feature": feature, "f_value": f_val, "p_value": p_val}
                )

                plt.figure(figsize=(10, 6))
                ax = sns.boxplot(x="target", y=feature, data=df)

                text_bbox_props = dict(
                    facecolor="lightgray", alpha=0.7, edgecolor="none", boxstyle="round"
                )

                # Calculate and display percentage difference and Cohen's d
                n_groups = len(groups)
                if n_groups == 2:
                    mean_group0 = np.mean(groups[0])
                    mean_group1 = np.mean(groups[1])
                    percentage_difference = (
                        ((mean_group1 - mean_group0) / mean_group0) * 100
                        if mean_group0 != 0
                        else np.inf
                    )
                    cohen_d = (mean_group1 - mean_group0) / np.sqrt(
                        (
                            np.std(groups[0], ddof=1) ** 2
                            + np.std(groups[1], ddof=1) ** 2
                        )
                        / 2
                    )

                    ax.text(
                        0.5,
                        0.95,
                        f"Percentage Difference: {abs(percentage_difference):.2f}%",
                        transform=ax.transAxes,
                        ha="center",
                        bbox=text_bbox_props,
                    )
                    ax.text(
                        0.5,
                        0.90,
                        f"Cohen's d: {abs(cohen_d):.2f}",
                        transform=ax.transAxes,
                        ha="center",
                        bbox=text_bbox_props,
                    )
                elif n_groups > 2:
                    pairwise_percentage_differences = []
                    pairwise_cohens_d = []

                    for (i, group_i), (j, group_j) in combinations(
                        enumerate(groups), 2
                    ):
                        mean_i = np.mean(group_i)
                        mean_j = np.mean(group_j)
                        percentage_difference = (
                            ((mean_j - mean_i) / mean_i) * 100
                            if mean_i != 0
                            else np.inf
                        )
                        cohen_d = (mean_j - mean_i) / np.sqrt(
                            (
                                np.std(group_i, ddof=1) ** 2
                                + np.std(group_j, ddof=1) ** 2
                            )
                            / 2
                        )

                        pairwise_percentage_differences.append(percentage_difference)
                        pairwise_cohens_d.append(cohen_d)

                    avg_percentage_difference = (
                        np.mean(pairwise_percentage_differences)
                        if pairwise_percentage_differences
                        else 0
                    )
                    avg_cohen_d = np.mean(pairwise_cohens_d) if pairwise_cohens_d else 0
                    cohen_d_interpretation = interpret_cohens_d(avg_cohen_d)

                    ax.text(
                        0.5,
                        0.95,
                        f"Avg. Percentage Difference: {abs(avg_percentage_difference):.2f}%",
                        transform=ax.transAxes,
                        ha="center",
                        bbox=text_bbox_props,
                    )
                    ax.text(
                        0.5,
                        0.90,
                        f"Avg. Cohen's d: {abs(avg_cohen_d):.2f} {cohen_d_interpretation}",
                        transform=ax.transAxes,
                        ha="center",
                        bbox=text_bbox_props,
                    )

                if p_val < 0.0001:
                    p_text = f"ANOVA p-value: {p_val:.2e}"
                else:
                    p_text = f"ANOVA p-value: {p_val:.4f}"

                plt.title(f"Distribution of {feature} by Target Group\n{p_text}")
                plt.xlabel("Target Class")
                plt.ylabel(feature)

                plot_filename = os.path.join(plot_dir, f"{feature}_boxplot.png")
                plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
                plt.close()
        return anova_results

    def select_best_rolling_window_for_each_feature(
        self, df, plot_dir="plots", rolling_windows=[24, 48, 72, 96, 120]
    ):
        if "target" not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")

        os.makedirs(plot_dir, exist_ok=True)

        features = [f for f in ALL_ON_CHAIN_FEATURES if f in df.columns]

        # Store results for each (feature, window) combination
        all_anova_results = []

        for window in rolling_windows:
            df_rolled = df.copy()

            # Apply rolling window mean per feature (grouped by sample id if needed)
            for feature in features:
                df_rolled[f"{feature}_roll{window}"] = (
                    df[feature].rolling(window=window, min_periods=1).mean()
                )

            # Run ANOVA on each rolled feature
            for feature in features:
                feature_roll = f"{feature}_roll{window}"
                if feature_roll not in df_rolled.columns:
                    continue

                groups = []
                for target_value in sorted(df_rolled["target"].dropna().unique()):
                    values = df_rolled[df_rolled["target"] == target_value][
                        feature_roll
                    ].dropna()
                    if not values.empty:
                        groups.append(values)

                if len(groups) > 1:  # Ensure we have at least 2 groups to compare
                    f_val, p_val = f_oneway(*groups)
                    all_anova_results.append(
                        {
                            "feature": feature,
                            "window": window,
                            "F-value": f_val,
                            "p-value": p_val,
                        }
                    )

        # Save ANOVA results for all rolling windows
        anova_df = pd.DataFrame(all_anova_results)
        anova_df.sort_values(["feature", "p-value"], inplace=True)
        anova_df.to_csv(os.path.join(plot_dir, "anova_window_results.csv"), index=False)

        # Keep only the best (lowest p-value) row for each feature
        best_anova_df = anova_df.loc[anova_df.groupby("feature")["p-value"].idxmin()]
        # Sort by p-value ascending
        best_anova_df.sort_values("p-value", inplace=True)

        # Find the best (lowest p-value) window per feature
        best_windows_df = anova_df.loc[
            anova_df.groupby("feature")["p-value"].idxmin()
        ].copy()

        # Sort by p-value or alphabetically
        best_windows_df.sort_values("p-value", inplace=True)

        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x="feature", y="window", data=best_windows_df, palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.title("Best Rolling Window Size by Feature (Lowest ANOVA p-value)")
        plt.ylabel("Window Size")
        plt.xlabel("Feature")
        plt.tight_layout()
        plt.grid(True, axis="y")
        plt.savefig(
            os.path.join(plot_dir, "anova_window_results_plot.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        return anova_df

    def select_best_rolling_window_for_all_features(self, df, feature_list):
        """
        Given an ANOVA result dataframe and a list of features,
        return the most common best window size.
        """
        # Filter to selected features
        df_selected = df[df["feature"].isin(feature_list)]

        # Get best (lowest p-value) window for each feature
        best_per_feature = df_selected.loc[
            df_selected.groupby("feature")["p-value"].idxmin()
        ]

        # Count the most common best window size
        most_common_window = best_per_feature["window"].value_counts().idxmax()

        return most_common_window


if __name__ == "__main__":
    dataset_path = "experimental/datasets/btc_data_with_target_latest_v2.csv"
    df_train, df_test = load_csv(dataset_path)
    feature_selection_class = FeatureSelecion()
    anova_results = feature_selection_class.prepare_features(df_train, plot_dir="plots")
    print("\nANOVA Results (sorted by p-value):")
    print(anova_results)
    anova_results.to_csv("anova_results.csv")

    best_common_window = feature_selection_class.select_best_common_window(
        df_train, ASSUMPTION_10
    )
    print("\nBest Window Size across assumption 10:")
    print(anova_results)
