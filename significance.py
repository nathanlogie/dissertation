import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv("combined_results.csv")


metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score",
           "Disparate Impact", "Statistical Parity Difference",
           "Average Odds Difference", "Equal Opportunity Difference"]


for dataset in df["Dataset"].unique():
    curr = df[df["Dataset"] == dataset]
    results = {}
    print(dataset)
    for metric in metrics:
        adaptive = curr[curr["Run Type"] == "Adaptive Masking"][metric]
        baseline = curr[curr["Run Type"] == "Baseline"][metric]

        t_stat, p_value = ttest_ind(adaptive, baseline, equal_var=False)

        significance = "Significant" if p_value < 0.05 else "Insignificant"
        results[metric] = significance

    for metric, significance in results.items():
        print(f"{metric}: {significance}")
