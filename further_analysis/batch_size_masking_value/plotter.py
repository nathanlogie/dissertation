#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def main():
    plt.style.use('seaborn-v0_8-white')

    df = pd.read_csv("significance_variance.csv")

    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

    # Iterate over each model (each row)
    for row_idx, model in enumerate(models):
        datasets = sorted(df[df["model"] == model]["dataset"].unique())
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            subset = df[(df["model"] == model) & (df["dataset"] == dataset)]
            subset = subset.sort_values("batchsize")
            ax.plot(subset["batchsize"], subset["variance"], marker='o',
                    linestyle='-', linewidth=2, markersize=6)

            display_dataset = "COMPAS" if dataset.strip().lower() == "recidivism compass" else dataset
            ax.set_title(display_dataset + f" (Model: {model})", fontsize=14)
            ax.set_xlabel("Batch Size", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"Variance", fontsize=12)
            else:
                ax.set_ylabel("Variance", fontsize=12)
            ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("batch_size_variance.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
