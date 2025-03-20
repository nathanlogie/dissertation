import pandas as pd
import numpy as np

df = pd.read_csv("batch_size_epoch_results.csv")

metrics = ["Accuracy", "Bal. Acc.", "Prec.", "Rec.", "F1 Score",
           "Disp. Impact", "Stat. Parity Diff.",
           "Avg. Odds Diff.", "Eq. Opp. Diff."]

results = []

for model in df["Model"].unique():
    for dataset in df[df["Model"] == model]["Dataset"].unique():
        subset = df[(df["Model"] == model) & (df["Dataset"] == dataset)]
        for batch in subset["Batch Size"].unique():
            sub_batch = subset[subset["Batch Size"] == batch]
            groups = {mv: sub_batch[sub_batch["Masking Value"] == mv] for mv in [-1, 0, 1]}

            metric_variances = []
            if all(len(groups[mv]) > 0 for mv in [-1, 0, 1]):
                for metric in metrics:
                    values = [groups[mv][metric].mean() for mv in [-1, 0, 1]]
                    var_val = np.var(values, ddof=1)
                    metric_variances.append(var_val)

                combined_variance = np.sum(metric_variances)
            else:
                combined_variance = np.nan

            results.append({
                "model": model,
                "dataset": dataset,
                "batchsize": batch,
                "variance": combined_variance
            })

output_df = pd.DataFrame(results)
output_df.to_csv("significance_variance.csv", index=False)
print("CSV file 'significance_variance.csv' has been created.")
