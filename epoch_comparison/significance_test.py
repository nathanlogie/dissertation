import pandas as pd
from scipy.stats import f_oneway


df = pd.read_csv("batch_size_results.csv")
metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score",
           "Disparate Impact", "Statistical Parity Difference",
           "Average Odds Difference", "Equal Opportunity Difference"]


results = {}

for metric in metrics:
    groups = [df[df["Batch Size"] == val][metric] for val in df["Batch Size"].unique()]
    f_stat, p_value = f_oneway(*groups)
    significance = "Significant" if p_value < 0.05 else "Insignificant"
    results[metric] = significance

# Print results
for metric, significance in results.items():
    print(f"{metric}: {significance}")
