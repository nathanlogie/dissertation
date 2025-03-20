import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

results_df = pd.read_csv("batch_size_epoch_results.csv")

metric_col = "metric"

datasets = results_df["Dataset"].unique()

fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)

if len(datasets) == 1:
    axes = [axes]

for ax, dataset in zip(axes, datasets):
    df_dataset = results_df[results_df["Dataset"] == dataset]
    batch_sizes = sorted(df_dataset["Batch Size"].unique())
    p_values = []

    for batch in batch_sizes:
        df_batch = df_dataset[df_dataset["Batch Size"] == batch]
        groups = []
        for mask in sorted(df_batch["Masking Value"].unique()):
            group_values = df_batch[df_batch["Masking Value"] == mask][metric_col].dropna().values
            groups.append(group_values)
        if len(groups) == 3:
            stat, p_val = f_oneway(*groups)
        else:
            p_val = None
        p_values.append(p_val)

    ax.plot(batch_sizes, p_values, marker='o', linestyle='-')
    ax.set_title(f"{dataset}")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("ANOVA p-value")
    ax.axhline(y=0.10, color='red', linestyle='--', label='p = 0.10')
    ax.legend()

plt.tight_layout()
plt.show()
