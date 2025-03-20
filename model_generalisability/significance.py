import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv("model_generalisability.csv")

metrics = ["Accuracy", "Bal. Acc.", "Prec.", "Rec.", "F1 Score",
           "Disp. Impact", "Stat. Parity Diff.",
           "Avg. Odds Diff.", "Eq. Opp. Diff."]

counter = {metric: 0 for metric in metrics}

for model in df["Model"].unique():
    for dataset in df[df["Model"] == model]["Dataset"].unique():
        curr = df[(df["Model"] == model) & (df["Dataset"] == dataset)]
        results = {}
        for metric in metrics:
            adaptive = curr[curr["Run Type"] == "Adaptive Masking"][metric]
            baseline = curr[curr["Run Type"] == "Baseline"][metric]

            t_stat, p_value = ttest_ind(adaptive, baseline, equal_var=False)

            significance = "Significant" if p_value <= 0.10 else "Insignificant"
            if significance == "Significant":
                print(f"{metric}: {p_value} : {dataset} : {model}")
                counter[metric] += 1

            results[metric] = significance



for metric, count in counter.items():
    print(f"{metric}: {count}")