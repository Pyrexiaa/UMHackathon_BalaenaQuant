from sample_data import BTC_DATA
from quantpilot.strategy import MLStrategy
import pandas as pd
from quantpilot.permutation_backtester import BacktesterPermutationTest
from quantpilot.models import get_model
import seaborn as sns
import matplotlib.pyplot as plt

# runner = BacktesterPermutationTest(
#     data=BTC_DATA,
#     strategy=MLStrategy(get_model("TCN")),
#     threshold_values=[0.3, 0.4, 0.5],
# )

# runner.run_all()
# runner.plot_permutation_heatmap(
#     metric="Sharpe Ratio",
#     show_plot=True,
# )


# Accumulate all strategy-permutation results
all_heatmaps = []

for strategy in ["TCN", "XGBoost"]:
    print(f"Running permutation test for strategy: {strategy}")
    runner = BacktesterPermutationTest(
        data=BTC_DATA,
        strategy=MLStrategy(get_model(strategy)),
        strategy_name=strategy,
        threshold_values=[0.3, 0.4, 0.5],
    )

    runner.run_all()

    # Save individual + collect for combined plot
    df = runner.plot_permutation_heatmap(metric="Sharpe Ratio", show_plot=False)
    all_heatmaps.append(df)

# Combine into a single DataFrame
combined_heatmap_df = pd.concat(all_heatmaps, axis=1)

# Final joint heatmap (Threshold vs Strategy)
plt.figure(figsize=(8, 6))
sns.heatmap(combined_heatmap_df, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Permutation Heatmap for Strategies and Thresholds - Sharpe Ratio")
plt.xlabel("Strategy")
plt.ylabel("Threshold")
plt.tight_layout()
plt.savefig("output/combined_sharpe_ratio_heatmap.png")
plt.show()
