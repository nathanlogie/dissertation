import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv("combined_results.csv")


def tost_test(adaptive, baseline, epsilon=0.01):

    mean_diff = np.mean(adaptive) - np.mean(baseline)

    lower_t, lower_p = ttest_ind(adaptive, baseline, alternative='greater')
    upper_t, upper_p = ttest_ind(adaptive, baseline, alternative='less')

    lower_bound = mean_diff - epsilon
    upper_bound = mean_diff + epsilon

    equiv = (lower_p < 0.05 and upper_p < 0.05)
    return equiv, lower_p, upper_p

metrics = ["Accuracy", "Bal. Acc.", "Prec.", "Rec.", "F1 Score",
           "Disp. Impact", "Stat. Parity Diff.",
           "Avg. Odds Diff.", "Eq. Opp. Diff."]

counter = {metric : 0 for metric in metrics}
tost_counter = {metric : 0 for metric in metrics}
for model in df["Model"].unique():
    curr = df[df["Model"] == model]
    for dataset in df["Dataset"].unique():
        curr = df[df["Dataset"] == dataset]
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

            # toster, _, _ = tost_test(adaptive, baseline)
            # if toster:
            #     print(f"{metric}: {p_value} : {dataset} : {model}")
            #     tost_counter[metric] += 1
for metric, count in counter.items():
    print(f"{metric}: {count}")