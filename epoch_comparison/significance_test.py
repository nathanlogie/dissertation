import pandas as pd
from scipy.stats import f_oneway


df = pd.read_csv("batch_size_results.csv")
metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score",
           "Disparate Impact", "Statistical Parity Difference",
           "Average Odds Difference", "Equal Opportunity Difference"]


results = {}
counter = {metric : 0 for metric in metrics}

for dataset in df["Dataset"].unique():
    curr = df[df["Dataset"] == dataset]
    results = {}
    print(dataset)
    for metric in metrics:
        groups = [df[df["Batch Size"] == val][metric] for val in df["Batch Size"].unique()]
        f_stat, p_value = f_oneway(*groups)
        significance = "Significant" if p_value < 0.05 else "Insignificant"
        results[metric] = significance
        if significance == "Significant":
            counter[metric] += 1

# Print results
for metric, count in counter.items():
    print(f"{metric}: {count}")
