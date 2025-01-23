import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv("simplified_combined_results.csv")


metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Disparate Impact",
           "Statistical Parity Difference", "PPV Parity", "FPR Parity"]

results = {}

for metric in metrics:
    adaptive = df[df["Run Type"] == "Adaptive Masking"][metric]
    baseline = df[df["Run Type"] == "Baseline"][metric]

    t_stat, p_value = ttest_ind(adaptive, baseline, equal_var=False)

    significance = "Significant" if p_value < 0.05 else "Insignificant"
    results[metric] = significance

for metric, significance in results.items():
    print(f"{metric}: {significance}")
