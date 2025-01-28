import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv("combined_results.csv")


metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score",
           "Disparate Impact", "Statistical Parity Difference",
           "Average Odds Difference", "Equal Opportunity Difference"]

counter = {metric : 0 for metric in metrics}

for dataset in df["Dataset"].unique():
    curr = df[df["Dataset"] == dataset]
    results = {}
    print(dataset)
    for metric in metrics:
        adaptive = curr[curr["Run Type"] == "Adaptive Masking"][metric]
        baseline = curr[curr["Run Type"] == "Baseline"][metric]

        t_stat, p_value = ttest_ind(adaptive, baseline, equal_var=False)

        significance = "Significant" if p_value < 0.10 else "Insignificant"
        results[metric] = significance
        if significance == "Significant":
            counter[metric] += 1

for metric, count in counter.items():
    print(f"{metric}: {count}")