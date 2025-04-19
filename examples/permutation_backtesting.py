from sample_data import BTC_DATA
from quantpilot.strategy import MLStrategy
import pandas as pd
from quantpilot.permutation_backtester import PermutationBacktester
from quantpilot.models import get_model
import seaborn as sns
import matplotlib.pyplot as plt

all_heatmaps = []

for strategy in [
    "TCN", 
    "XGBoost"
    ]:
    print(f"Running permutation test for strategy: {strategy}")
    runner = PermutationBacktester(
        data=BTC_DATA,
        strategy=MLStrategy(get_model(strategy)),
        strategy_name=strategy,
        threshold_values=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        mode="geometric",
        entry_exit_logic="mean_reversion",
    )

    runner.run_all(
        backtest_end_date="2023-12-31"
    )  # Run backtest only

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
